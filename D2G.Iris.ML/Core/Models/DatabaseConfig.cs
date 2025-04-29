namespace D2G.Iris.ML.Core.Models
{
    public class DatabaseConfig
    {
        public string Server { get; set; }
        public string Database { get; set; }
        public string TableName { get; set; }
        public string OutputTableName { get; set; }
        public string WhereClause { get; set; }
    }
}