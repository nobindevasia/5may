using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using Microsoft.ML;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface ISqlHandler
    {
        void Connect(DatabaseConfig dbConfig);
        string GetConnectionString();
        void SaveToSql(
            string tableName,
            IDataView processedData,
            string[] featureNames,
            string targetField,
            ModelType modelType);
     
    }
}
