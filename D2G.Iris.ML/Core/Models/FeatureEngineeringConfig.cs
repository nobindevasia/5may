using D2G.Iris.ML.Core.Enums;


namespace D2G.Iris.ML.Core.Models
{
    public class FeatureEngineeringConfig
    {
        public FeatureSelectionMethod Method { get; set; }
        public int ExecutionOrder { get; set; }
        public int NumberOfComponents { get; set; }
        public int MaxFeatures { get; set; }
        public double MinImprovement { get; set; }
        public double MulticollinearityThreshold { get; set; }


    }
}