using D2G.Iris.ML.Core.Models;
using Microsoft.ML;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataProcessor
    {
        Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config);
    }
}