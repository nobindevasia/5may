using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Data
{
    public class DatabaseDataLoader : IDataLoader
    {
        private readonly MLContext _mlContext;
        private long? _lastLoadedRowCount;

        public DatabaseDataLoader(MLContext mlContext = null)
        {
            _mlContext = mlContext ?? new MLContext();
        }

        public IDataView LoadDataFromSql(
            string sqlConnectionString,
            string tableName,
            IEnumerable<string> featureColumns,
            ModelType modelType,
            string targetColumn,
            string whereSyntax = "")
        {
            Console.WriteLine("=============== Loading Data ===============");

            string fullTableName = tableName.Contains('.')
                ? string.Join('.', tableName.Split('.').Select(part => $"[{part}]"))
                : $"[{tableName}]";

            using (var conn = new SqlConnection(sqlConnectionString))
            {
                conn.Open();
                var countSql = $"SELECT COUNT(*) FROM {fullTableName}" +
                               (!string.IsNullOrWhiteSpace(whereSyntax)
                                    ? $" WHERE {whereSyntax}" : string.Empty);
                using (var countCmd = new SqlCommand(countSql, conn))
                {
                    _lastLoadedRowCount = Convert.ToInt64(countCmd.ExecuteScalar());
                }
            }

            var allCols = featureColumns.Concat(new[] { targetColumn })
                                        .Select(c => $"[{c}]");
            var sql = $"SELECT {string.Join(", ", allCols)} FROM {fullTableName}" +
                      (!string.IsNullOrWhiteSpace(whereSyntax)
                            ? $" WHERE {whereSyntax}" : string.Empty);

            var loaderCols = new List<DatabaseLoader.Column>();
            int idx = 0;
            foreach (var feat in featureColumns)
            {
                loaderCols.Add(new DatabaseLoader.Column(
                    name: feat,
                    dbType: DbType.Single,
                    index: idx++
                ));
            }

            DbType labelDbType = modelType switch
            {
                ModelType.BinaryClassification => DbType.Int64,
                ModelType.MultiClassClassification => DbType.Int64,
                ModelType.Regression => DbType.Single,
                _ => DbType.Int64
            };
            loaderCols.Add(new DatabaseLoader.Column(
                name: targetColumn,
                dbType: labelDbType,
                index: idx
            ));
            var dbLoader = _mlContext.Data.CreateDatabaseLoader(loaderCols.ToArray());
            var dbSource = new DatabaseSource(
                providerFactory: SqlClientFactory.Instance,
                connectionString: sqlConnectionString,
                commandText: sql
            );

            var dataView = dbLoader.Load(dbSource);
            Console.WriteLine($">> Loaded {_lastLoadedRowCount ?? 0} rows of data.");
            return dataView;
        }

      
    }
}
