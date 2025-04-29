using System;
using System.Data;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Interfaces;

namespace D2G.Iris.ML.Data
{
    public class SqlHandler : ISqlHandler
    {
        private SqlConnectionStringBuilder _builder;
        private readonly string _tableName;

        public SqlHandler(string tableName)
        {
            _tableName = tableName ?? throw new ArgumentNullException(nameof(tableName));
        }

        public void Connect(DatabaseConfig dbConfig)
        {
            _builder = new SqlConnectionStringBuilder
            {
                DataSource = dbConfig.Server,
                InitialCatalog = dbConfig.Database,
                IntegratedSecurity = true,
                TrustServerCertificate = true,
                ConnectTimeout = 60
            };
        }

        public string GetConnectionString()
        {
            if (_builder == null)
                throw new InvalidOperationException("Database connection not initialized.");
            return _builder.ConnectionString;
        }

        public void SaveToSql(
            string tableName,
            IDataView data,
            string[] featureNames,
            string targetField,
            ModelType modelType)
        {
            var destTable = string.IsNullOrWhiteSpace(tableName) ? _tableName : tableName;
            if (string.IsNullOrWhiteSpace(destTable))
                throw new ArgumentException("Table name must be provided.", nameof(tableName));

            var dataTable = new DataTable();
            foreach (var feature in featureNames)
                dataTable.Columns.Add(feature, typeof(float));

            Type targetDotNetType = modelType switch
            {
                ModelType.BinaryClassification => typeof(bool),
                ModelType.MultiClassClassification => typeof(int),
                _ => typeof(float)
            };
            dataTable.Columns.Add(targetField, targetDotNetType);

            var featsCol = data.Schema.GetColumnOrNull("Features");
            bool isPca = featsCol.HasValue && featureNames.All(f => f.StartsWith("PCA_Component_"));

            var featureRows = new List<float[]>();
            int rowCount;

            if (isPca)
            {
                var vectors = data.GetColumn<VBuffer<float>>("Features").ToArray();
                rowCount = vectors.Length;
                foreach (var vec in vectors)
                {
                    var vals = vec.GetValues().ToArray();
                    var rowVec = new float[featureNames.Length];
                    for (int j = 0; j < featureNames.Length; j++)
                        rowVec[j] = j < vals.Length ? vals[j] : 0f;
                    featureRows.Add(rowVec);
                }
            }
            else
            {
                var columns = new float[featureNames.Length][];
                for (int i = 0; i < featureNames.Length; i++)
                    columns[i] = data.GetColumn<float>(featureNames[i]).ToArray();

                rowCount = columns.Any() ? columns[0].Length : 0;
                for (int r = 0; r < rowCount; r++)
                {
                    var rowVec = new float[featureNames.Length];
                    for (int c = 0; c < featureNames.Length; c++)
                        rowVec[c] = columns[c][r];
                    featureRows.Add(rowVec);
                }
            }

            var colOpt = data.Schema.GetColumnOrNull(targetField);
            if (!colOpt.HasValue)
                throw new InvalidOperationException($"Target column '{targetField}' not found in schema.");
            var rawType = colOpt.Value.Type;
            Array targetArray;

            if (modelType == ModelType.BinaryClassification)
            {
                if (rawType is BooleanDataViewType)
                    targetArray = data.GetColumn<bool>(targetField).ToArray();
                else if (rawType is NumberDataViewType numLong && numLong.RawType == typeof(long))
                    targetArray = data.GetColumn<long>(targetField).Select(l => l > 0).ToArray();
                else if (rawType is NumberDataViewType numFloat && numFloat.RawType == typeof(float))
                    targetArray = data.GetColumn<float>(targetField).Select(f => f > 0).ToArray();
                else
                    throw new InvalidOperationException($"Cannot convert target column '{targetField}' of type {rawType} to bool.");
            }
            else if (modelType == ModelType.MultiClassClassification)
            {
                if (rawType is NumberDataViewType numLong && numLong.RawType == typeof(long))
                    targetArray = data.GetColumn<long>(targetField).Select(l => (int)l).ToArray();
                else if (rawType is NumberDataViewType numFloat && numFloat.RawType == typeof(float))
                    targetArray = data.GetColumn<float>(targetField).Select(f => Convert.ToInt32(f)).ToArray();
                else if (rawType is BooleanDataViewType)
                    targetArray = data.GetColumn<bool>(targetField).Select(b => b ? 1 : 0).ToArray();
                else
                    throw new InvalidOperationException($"Cannot convert target column '{targetField}' of type {rawType} to int.");
            }
            else // Regression
            {
                if (rawType is NumberDataViewType numFloat && numFloat.RawType == typeof(float))
                    targetArray = data.GetColumn<float>(targetField).ToArray();
                else if (rawType is NumberDataViewType numLong && numLong.RawType == typeof(long))
                    targetArray = data.GetColumn<long>(targetField).Select(l => Convert.ToSingle(l)).ToArray();
                else if (rawType is BooleanDataViewType)
                    targetArray = data.GetColumn<bool>(targetField).Select(b => b ? 1f : 0f).ToArray();
                else
                    throw new InvalidOperationException($"Cannot convert target column '{targetField}' of type {rawType} to float.");
            }


            for (int r = 0; r < rowCount; r++)
            {
                var dr = dataTable.NewRow();
                var rowVec = featureRows[r];
                for (int c = 0; c < featureNames.Length; c++)
                    dr[featureNames[c]] = rowVec[c];
                dr[targetField] = targetArray.GetValue(r);
                dataTable.Rows.Add(dr);
            }


            using var connection = new SqlConnection(GetConnectionString());
            connection.Open();
            string colsDef = string.Join(", ", featureNames.Select(f => $"[{f}] FLOAT"));
            string targetSqlType = modelType switch
            {
                ModelType.BinaryClassification => "BIT",
                ModelType.MultiClassClassification => "INT",
                _ => "FLOAT"
            };
            string createSql = $@"
IF OBJECT_ID('{destTable}', 'U') IS NOT NULL
    DROP TABLE {destTable};
CREATE TABLE {destTable} (
    {colsDef},
    [{targetField}] {targetSqlType},
    ProcessedDateTime DATETIME DEFAULT GETDATE()
);";
            using var cmd = new SqlCommand(createSql, connection);
            cmd.ExecuteNonQuery();

            using var bulk = new SqlBulkCopy(connection)
            {
                DestinationTableName = destTable,
                BatchSize = 1000,
                BulkCopyTimeout = 300
            };
            foreach (var feature in featureNames)
                bulk.ColumnMappings.Add(feature, feature);
            bulk.ColumnMappings.Add(targetField, targetField);
            bulk.WriteToServer(dataTable);
        }
    }
}
