using System.Collections.Generic;

namespace D2G.Iris.ML.Core.Models
{
    public class TrainingParameters
    {
        public string Algorithm { get; set; }
        public Dictionary<string, object> AlgorithmParameters { get; set; }
        public double TestFraction { get; set; }
    }
}