using System;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class NoFeatureSelector : BaseFeatureSelector
    {
        public NoFeatureSelector(MLContext mlContext)
            : base(mlContext)
        {
        }

        public override Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            InitializeReport("No");
            _report.AppendLine($"Using all enabled features: {candidateFeatures.Length}");

            foreach (var feature in candidateFeatures)
            {
                _report.AppendLine($"- {feature}");
            }

            var transformedData = CreateFeaturesColumn(data, candidateFeatures);

            AddFeatureSelectionSummary(
                candidateFeatures.Length,
                candidateFeatures.Length,
                candidateFeatures);

            return Task.FromResult((transformedData, candidateFeatures, _report.ToString()));
        }
    }
}