using D2G.Iris.ML.Core.Enums;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataLoader
    {
        IDataView LoadDataFromSql(
            string sqlConnectionString,
            string tableName,
            IEnumerable<string> featureColumns,
            ModelType modelType,
            string targetColumn,
            string whereSyntax = "");
    }
}
