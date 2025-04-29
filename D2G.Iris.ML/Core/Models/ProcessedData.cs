using System.Collections.Generic;
using D2G.Iris.ML.Core.Enums;
using Microsoft.ML;


namespace D2G.Iris.ML.Core.Models
{
    public class ProcessedData
    {
        public IDataView Data { get; set; }
        public string[] FeatureNames { get; set; }
        public int OriginalSampleCount { get; set; }
        public int BalancedSampleCount { get; set; }
        public string FeatureSelectionReport { get; set; }
        public FeatureSelectionMethod FeatureSelectionMethod { get; set; }
        public DataBalanceMethod DataBalancingMethod { get; set; }
        public int DataBalancingExecutionOrder { get; set; }
        public int FeatureSelectionExecutionOrder { get; set; }
    }
}