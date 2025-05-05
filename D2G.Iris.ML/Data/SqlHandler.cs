using System;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Data
{
    public class SqlHandler : ISqlHandler
    {
        private readonly string _defaultTableName;
        private readonly MLContext _mlContext;
        private SqlConnectionStringBuilder _builder;

        public SqlHandler(string defaultTableName, MLContext mlContext = null)
        {
            _defaultTableName = defaultTableName
                ?? throw new ArgumentNullException(nameof(defaultTableName));
            _mlContext = mlContext ?? new MLContext();
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

        /// <summary>
        /// Saves every row in <paramref name="data"/>—projected as T—into SQL via bulk-copy.
        /// T must have public properties for each feature + target field.
        /// </summary>
        public void SaveToSql<T>(IDataView data, string tableName = null) where T : class, new()
        {
            // determine destination table
            var destTable = string.IsNullOrWhiteSpace(tableName)
                              ? _defaultTableName
                              : tableName;
            if (string.IsNullOrWhiteSpace(destTable))
                throw new ArgumentException("Table name must be provided.", nameof(tableName));

            // project IDataView → IEnumerable<T>
            var rows = _mlContext.Data
                                 .CreateEnumerable<T>(data, reuseRowObject: false)
                                 .ToList();
            if (!rows.Any())
                return;  // nothing to save

            // build a DataTable via reflection
            var dataTable = ToDataTable(rows);

            // (re)create the SQL table
            var createSql = BuildCreateTableSql<T>(destTable);
            using var conn = new SqlConnection(GetConnectionString());
            conn.Open();
            using (var cmd = new SqlCommand(createSql, conn))
                cmd.ExecuteNonQuery();

            // bulk-copy
            using var bulk = new SqlBulkCopy(conn)
            {
                DestinationTableName = destTable,
                BatchSize = 1_000,
                BulkCopyTimeout = 300
            };
            foreach (DataColumn col in dataTable.Columns)
                bulk.ColumnMappings.Add(col.ColumnName, col.ColumnName);

            bulk.WriteToServer(dataTable);
        }

        // helper: convert list of POCOs to DataTable
        private static DataTable ToDataTable<T>(IEnumerable<T> data)
        {
            var table = new DataTable();
            var props = typeof(T).GetProperties();

            // add columns
            foreach (var p in props)
            {
                var type = Nullable.GetUnderlyingType(p.PropertyType)
                           ?? p.PropertyType;
                table.Columns.Add(p.Name, type);
            }

            // add rows
            foreach (var item in data)
            {
                var values = props
                  .Select(p => p.GetValue(item) ?? DBNull.Value)
                  .ToArray();
                table.Rows.Add(values);
            }

            return table;
        }

        // helper: emits a DROP+CREATE DDL that matches T’s properties to SQL types
        private static string BuildCreateTableSql<T>(string tableName)
        {
            var cols = typeof(T)
              .GetProperties()
              .Select(p =>
              {
                  var t = Nullable.GetUnderlyingType(p.PropertyType)
                          ?? p.PropertyType;
                  var sqlType = t == typeof(bool) ? "BIT"
                              : t == typeof(int) ? "INT"
                              : t == typeof(long) ? "BIGINT"
                              : t == typeof(DateTime) ? "DATETIME"
                              : "FLOAT";
                  return $"[{p.Name}] {sqlType}";
              });

            return $@"
IF OBJECT_ID('{tableName}', 'U') IS NOT NULL
    DROP TABLE {tableName};
CREATE TABLE {tableName} (
    {string.Join(", ", cols)},
    ProcessedDateTime DATETIME DEFAULT GETDATE()
);";
        }
    }
}
