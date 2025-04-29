using System;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.DataBalancing
{
    public class NoDataBalancer : IDataBalancer
    {
        public Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            Console.WriteLine("No data balancing applied - returning original dataset");
            return Task.FromResult(data);
        }
    }
}