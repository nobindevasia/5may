using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Training
{
    public class TrainerFactory
    {
        private readonly MLContext _mlContext;

        public TrainerFactory(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public IEstimator<ITransformer> GetTrainer(ModelType modelType, TrainingParameters parameters)
        {
            return modelType switch
            {
                ModelType.BinaryClassification => GetBinaryClassificationTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                ModelType.MultiClassClassification => GetMultiClassClassificationTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                ModelType.Regression => GetRegressionTrainer(parameters.Algorithm, parameters.AlgorithmParameters),
                _ => throw new ArgumentException($"Unsupported model type: {modelType}")
            };
        }

        private IEstimator<ITransformer> GetBinaryClassificationTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            return algorithm.ToLower() switch
            {
                "fastforest" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FastForest,
                    new FastForestBinaryTrainer.Options(), parameters),

                "fasttree" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FastTree,
                    new FastTreeBinaryTrainer.Options(), parameters),

                "lightgbm" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LightGbm,
                    new LightGbmBinaryTrainer.Options(), parameters),

                "sdcalogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression,
                    new SdcaLogisticRegressionBinaryTrainer.Options(), parameters),

                "averagedperceptron" => CreateTrainer(_mlContext.BinaryClassification.Trainers.AveragedPerceptron,
                    new AveragedPerceptronTrainer.Options(), parameters),

               "linearSvm" => CreateTrainer (_mlContext.BinaryClassification.Trainers.LinearSvm,
                    new LinearSvmTrainer.Options(), parameters),

                "ldsvm" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LdSvm,
                    new LdSvmTrainer.Options(), parameters),

                "sdca" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SdcaNonCalibrated,
                    new SdcaNonCalibratedBinaryTrainer.Options(), parameters),

                "sgdcalibrated" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SgdCalibrated,
                    new SgdCalibratedTrainer.Options(), parameters),

                "symbolicsgdlogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression,
                    new SymbolicSgdLogisticRegressionBinaryTrainer.Options(), parameters),

                "gam" => CreateTrainer(_mlContext.BinaryClassification.Trainers.Gam,
                    new GamBinaryTrainer.Options(), parameters),

                "fieldawareFactorizationMachine" => CreateTrainer(_mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine,
                    new FieldAwareFactorizationMachineTrainer.Options(), parameters),

                "lbfgslogisticregression" => CreateTrainer(_mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression,
                    new LbfgsLogisticRegressionBinaryTrainer.Options(), parameters),


                _ => throw new ArgumentException($"Unsupported binary classification algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> GetMultiClassClassificationTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            return algorithm.ToLower() switch
            {
                "lightgbm" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.LightGbm,
                    new LightGbmMulticlassTrainer.Options(), parameters),

                "sdcamaximumentropy" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy,
                    new SdcaMaximumEntropyMulticlassTrainer.Options(), parameters),

                "sdca" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated,
                    new SdcaNonCalibratedMulticlassTrainer.Options(), parameters),

                "fasttree" => _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                            _mlContext.BinaryClassification.Trainers.FastTree(
                            new FastTreeBinaryTrainer.Options()
                            {
                                LabelColumnName = "Label",
                                FeatureColumnName = "Features"
                            })),

                "fastforest" => _mlContext.MulticlassClassification.Trainers.OneVersusAll(
                            _mlContext.BinaryClassification.Trainers.FastForest(
                            new FastForestBinaryTrainer.Options()
                            {
                                LabelColumnName = "Label",
                                FeatureColumnName = "Features"
                            })),

                "lbfgsmaximumentropy" => CreateTrainer(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy,
                                new LbfgsMaximumEntropyMulticlassTrainer.Options(), parameters),
                _ => throw new ArgumentException($"Unsupported multiclass classification algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> GetRegressionTrainer(string algorithm, Dictionary<string, object> parameters)
        {
            return algorithm.ToLower() switch
            {
                "fastforest" => CreateTrainer(_mlContext.Regression.Trainers.FastForest,
                    new FastForestRegressionTrainer.Options(), parameters),

                "fasttree" => CreateTrainer(_mlContext.Regression.Trainers.FastTree,
                    new FastTreeRegressionTrainer.Options(), parameters),

                "lightgbm" => CreateTrainer(_mlContext.Regression.Trainers.LightGbm,
                    new LightGbmRegressionTrainer.Options(), parameters),

                "ols" => CreateTrainer(_mlContext.Regression.Trainers.Ols,
                    new OlsTrainer.Options(), parameters),

                "onlinegradientdescent" => CreateTrainer(_mlContext.Regression.Trainers.OnlineGradientDescent,
                    new OnlineGradientDescentTrainer.Options(), parameters),

                "gam" => CreateTrainer(_mlContext.Regression.Trainers.Gam,
                    new GamRegressionTrainer.Options(), parameters),

                "sdca" => CreateTrainer(_mlContext.Regression.Trainers.Sdca,
                    new SdcaRegressionTrainer.Options(), parameters),

                "fasttreetweedie" => CreateTrainer(_mlContext.Regression.Trainers.FastTreeTweedie,
                    new FastTreeTweedieTrainer.Options(), parameters),

                "lbfgspoissonregression" => CreateTrainer(_mlContext.Regression.Trainers.LbfgsPoissonRegression,
                    new LbfgsPoissonRegressionTrainer.Options(), parameters),
                _ => throw new ArgumentException($"Unsupported regression algorithm: {algorithm}")
            };
        }

        private IEstimator<ITransformer> CreateTrainer<TOptions>(
            Func<TOptions, IEstimator<ITransformer>> trainerBuilder,
            TOptions options,
            Dictionary<string, object> parameters) where TOptions : class
        {
            options.GetType().GetProperty("LabelColumnName")?.SetValue(options, "Label");
            options.GetType().GetProperty("FeatureColumnName")?.SetValue(options, "Features");
            ApplyParameters(options, parameters);
            return trainerBuilder(options);
        }

        private static void ApplyParameters<T>(T options, Dictionary<string, object> parameters)
        {
            if (parameters == null) return;

            var type = typeof(T);
            var members = type.GetMembers(BindingFlags.Public | BindingFlags.Instance)
                .Where(m => m.MemberType == MemberTypes.Property || m.MemberType == MemberTypes.Field)
                .ToDictionary(m => m.Name, m => m);

            foreach (var (key, value) in parameters)
            {
                var memberName = char.ToUpper(key[0]) + key[1..];
                if (!members.TryGetValue(memberName, out var member))
                {
                    Console.WriteLine($"Warning: Parameter '{memberName}' is not recognized on {type.Name}.");
                    continue;
                }

                try
                {
                    var targetType = member switch
                    {
                        PropertyInfo prop => prop.PropertyType,
                        FieldInfo field => field.FieldType,
                        _ => throw new InvalidOperationException("Unexpected member type")
                    };

                    var convertedValue = value switch
                    {
                        JsonElement jsonElement => ConvertJsonElement(jsonElement, targetType),
                        _ => Convert.ChangeType(value, targetType)
                    };

                    switch (member)
                    {
                        case PropertyInfo prop:
                            prop.SetValue(options, convertedValue);
                            break;
                        case FieldInfo field:
                            field.SetValue(options, convertedValue);
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to set parameter '{key}'. Error: {ex.Message}");
                }
            }
        }

        private static object ConvertJsonElement(JsonElement element, Type targetType)
        {
            var nonNullableType = Nullable.GetUnderlyingType(targetType) ?? targetType;

            return nonNullableType switch
            {
                Type t when t == typeof(int) => element.GetInt32(),
                Type t when t == typeof(double) => element.GetDouble(),
                Type t when t == typeof(float) => element.GetSingle(),
                Type t when t == typeof(bool) => element.GetBoolean(),
                Type t when t == typeof(string) => element.GetString(),
                Type t when t.IsEnum => Enum.Parse(t, element.GetString(), true),
                _ => Convert.ChangeType(element.GetRawText(), nonNullableType)
            };
        }
    }
}