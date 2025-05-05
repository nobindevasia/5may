using System;
using System.Data;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

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
            {
                dataTable.Columns.Add(feature, typeof(float));
            }

            Type targetDotNetType = modelType switch
            {
                ModelType.BinaryClassification => typeof(long),
                ModelType.MultiClassClassification => typeof(long),
                _ => typeof(float)
            };
            dataTable.Columns.Add(targetField, targetDotNetType);

            var featuresCol = data.Schema.GetColumnOrNull("Features");
            var targetCol = data.Schema.GetColumnOrNull(targetField);

            if (!featuresCol.HasValue)
                throw new InvalidOperationException("Features column not found");
            if (!targetCol.HasValue)
                throw new InvalidOperationException($"Target column '{targetField}' not found");

            var cursor = data.GetRowCursor(data.Schema);
            var featuresGetter = cursor.GetGetter<VBuffer<float>>(featuresCol.Value);
            var targetType = targetCol.Value.Type;

            ValueGetter<float> targetFloatGetter = null;
            ValueGetter<long> targetLongGetter = null;
            ValueGetter<bool> targetBoolGetter = null;

            if (targetType is NumberDataViewType numType)
            {
                if (numType.RawType == typeof(long))
                    targetLongGetter = cursor.GetGetter<long>(targetCol.Value);
                else
                    targetFloatGetter = cursor.GetGetter<float>(targetCol.Value);
            }
            else if (targetType is BooleanDataViewType)
            {
                targetBoolGetter = cursor.GetGetter<bool>(targetCol.Value);
            }

            var featureBuffer = default(VBuffer<float>);
            while (cursor.MoveNext())
            {
                var row = dataTable.NewRow();

                featuresGetter(ref featureBuffer);
                var denseValues = featureBuffer.GetValues().ToArray();

                for (int i = 0; i < featureNames.Length && i < denseValues.Length; i++)
                {
                    row[featureNames[i]] = denseValues[i];
                }

                if (targetLongGetter != null)
                {
                    long val = 0;
                    targetLongGetter(ref val);
                    row[targetField] = val;
                }
                else if (targetFloatGetter != null)
                {
                    float val = 0;
                    targetFloatGetter(ref val);
                    row[targetField] = modelType == ModelType.BinaryClassification ?
                        (val > 0.5f ? 1L : 0L) : val;
                }
                else if (targetBoolGetter != null)
                {
                    bool val = false;
                    targetBoolGetter(ref val);
                    row[targetField] = modelType == ModelType.BinaryClassification ?
                        (val ? 1L : 0L) : (val ? 1f : 0f);
                }

                dataTable.Rows.Add(row);
            }

            using var connection = new SqlConnection(GetConnectionString());
            connection.Open();

            var columnDefinitions = featureNames
                .Select(f => $"[{f}] FLOAT")
                .Concat(new[]
                {
                    $"[{targetField}] {(modelType == ModelType.Regression ? "FLOAT" : "BIGINT")}",
                    "ProcessedDateTime DATETIME DEFAULT GETDATE()"
                });

            string createSql = $@"
IF OBJECT_ID('{destTable}', 'U') IS NOT NULL
    DROP TABLE {destTable};
CREATE TABLE {destTable} (
    {string.Join(",\n    ", columnDefinitions)}
);";

            using (var cmd = new SqlCommand(createSql, connection))
            {
                cmd.ExecuteNonQuery();
            }

            using var bulk = new SqlBulkCopy(connection)
            {
                DestinationTableName = destTable,
                BatchSize = 1000,
                BulkCopyTimeout = 300
            };

            foreach (var feature in featureNames)
            {
                bulk.ColumnMappings.Add(feature, feature);
            }
            bulk.ColumnMappings.Add(targetField, targetField);

            bulk.WriteToServer(dataTable);
        }
    }
}