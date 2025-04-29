using D2G.Iris.ML.Core.Models;
using Microsoft.ML;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataBalancer
    {
        Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField);
    }
}