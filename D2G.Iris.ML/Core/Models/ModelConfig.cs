using System.Collections.Generic;
using D2G.Iris.ML.Core.Enums;


namespace D2G.Iris.ML.Core.Models
{
    public class ModelConfig
    {
        public string Author { get; set; }
        public string Description { get; set; }
        public ModelType ModelType { get; set; }
        public List<InputField> InputFields { get; set; } = new List<InputField>();
        public TrainingParameters TrainingParameters { get; set; }
        public DatabaseConfig Database { get; set; }
        public FeatureEngineeringConfig FeatureEngineering { get; set; }
        public DataBalancingConfig DataBalancing { get; set; } = new DataBalancingConfig();
        public string TargetField { get; set; }

    }
}