using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Interfaces;
namespace D2G.Iris.ML.Core.Models
{
    public abstract class StandardizedBaseMetrics : IStandardizedBaseMetrics
    {
        public virtual string CreateStandardizedMetricsMsg()
        {
            return "No metrics available.";
        }
    }

    public class GeneralInfo
    {
        public string AuthorName { get; set; }
        public string Description { get; set; }
    }

    public class TrainingInfo
    {
        public string TrainerName { get; set; }
        public int NumberColumns { get; set; }
        public long NumberRows { get; set; }  
        public string[] InputColumnNames { get; set; }
        public string[] OutputColumnNames { get; set; }
    }

    public class StandardizedBinaryMetrics : StandardizedBaseMetrics
    {
        public double Accuracy { get; set; }
        public double AreaUnderRocCurve { get; set; }
        public double PositivePrecision { get; set; }
        public double PositiveRecall { get; set; }
        public double F1Score { get; set; }
        public double AreaUnderPrecisionRecallCurve { get; set; }




        public override string CreateStandardizedMetricsMsg()
        {
            return $"Binary Classification Metrics:\n" +
                   $"Accuracy: {Accuracy:F4}\n" +
                   $"AUC: {AreaUnderRocCurve:F4}\n" +
                   $"Precision: {PositivePrecision:F4}\n" +
                   $"Recall: {PositiveRecall:F4}\n" +
                   $"F1 Score: {F1Score:F4}\n"+
                   $"Area Under Precision-Recall Curve: {AreaUnderPrecisionRecallCurve:F4}";    
        }
    }

    public class StandardizedMulticlassMetrics : StandardizedBaseMetrics
    {
        public double MicroAccuracy { get; set; }
        public double MacroAccuracy { get; set; }
        public double LogLoss { get; set; }
        public double LogLossReduction { get; set; }

        public override string CreateStandardizedMetricsMsg()
        {
            return $"Multiclass Classification Metrics:\n" +
                   $"Micro Accuracy: {MicroAccuracy:F4}\n" +
                   $"Macro Accuracy: {MacroAccuracy:F4}\n" +
                   $"Log Loss: {LogLoss:F4}\n" +
                   $"Log Loss Reduction: {LogLossReduction:F4}";
        }
    }


   

    public class StandardizedRegressionMetrics : StandardizedBaseMetrics
    {
        public double RSquared { get; set; }
        public double MeanAbsoluteError { get; set; }
        public double MeanSquaredError { get; set; }
        public double RootMeanSquaredError { get; set; }

        public override string CreateStandardizedMetricsMsg()
        {
            return $"Regression Metrics:\n" +
                   $"R²: {RSquared:F4}\n" +
                   $"Mean Absolute Error: {MeanAbsoluteError:F4}\n" +
                   $"Mean Squared Error: {MeanSquaredError:F4}\n" +
                   $"Root Mean Squared Error: {RootMeanSquaredError:F4}";
        }

        
    }
    public class DefaultStandardizedMetrics : StandardizedBaseMetrics
    {

    }

}
