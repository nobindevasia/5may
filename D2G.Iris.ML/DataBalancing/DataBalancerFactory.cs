using System;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;

namespace D2G.Iris.ML.DataBalancing
{
    public class DataBalancerFactory
    {
        public IDataBalancer CreateBalancer(DataBalanceMethod method)
        {
            return method switch
            {
                DataBalanceMethod.None => new NoDataBalancer(),
                DataBalanceMethod.SMOTE => new SmoteDataBalancer(),
                _ => throw new ArgumentException($"Unsupported data balance method: {method}")
            };
        }
    }
}