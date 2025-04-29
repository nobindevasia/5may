using System;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.FeatureEngineering
{

    public abstract class BaseFeatureSelector : IFeatureSelector
    {
        protected readonly MLContext _mlContext;
        protected readonly StringBuilder _report;

        protected BaseFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
            _report = new StringBuilder();
        }


        public abstract Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config);

        protected IDataView CreateFeaturesColumn(IDataView data, string[] featureNames)
        {
            var pipeline = _mlContext.Transforms.Concatenate("Features", featureNames);
            return pipeline.Fit(data).Transform(data);
        }

        protected void InitializeReport(string methodName)
        {
            _report.Clear();
            _report.AppendLine($"\n{methodName} Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");
        }


        protected void AddFeatureSelectionSummary(
            int originalFeatureCount,
            int selectedFeatureCount,
            string[] selectedFeatures)
        {
            _report.AppendLine($"\nSelection Summary:");
            _report.AppendLine($"Original features: {originalFeatureCount}");
            _report.AppendLine($"Selected features: {selectedFeatureCount}");

            _report.AppendLine("\nSelected Features:");
            foreach (var feature in selectedFeatures)
            {
                _report.AppendLine($"- {feature}");
            }
        }

        protected void AddErrorToReport(Exception ex)
        {
            _report.AppendLine($"Error during feature selection: {ex.Message}");
            Console.WriteLine($"Full error details: {ex}");
        }

        protected virtual void ValidateConfiguration(FeatureEngineeringConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));
        }
    }
}