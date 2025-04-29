using D2G.Iris.ML.Core.Enums;


namespace D2G.Iris.ML.Core.Models
{
    public class DataBalancingConfig
    {
        public DataBalanceMethod Method { get; set; }
        public int ExecutionOrder { get; set; }
        public int KNeighbors { get; set; }
        public float UndersamplingRatio { get; set; }
        public float MinorityToMajorityRatio { get; set; }
    }
}