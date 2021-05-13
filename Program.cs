using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace AnomalyDetectionSample
{
    internal class Program
    {
        private static string targetFile = Path.Combine(Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName, "data.csv");

        private static void Main(string[] args)
        {
            var context = new MLContext();
            var data = context.Data.LoadFromTextFile<CurrencyModel>(targetFile, hasHeader: true, separatorChar: ',');

            Console.WriteLine("***************************\tANOMALY DETECTION\t***************************");

            var pipelineAnomaly = context.Transforms.DetectSpikeBySsa(nameof(SpikeAnomaly.Anomalies),
                nameof(CurrencyModel.Close), confidence: 98.0,
                trainingWindowSize: 90, seasonalityWindowSize: 30, pvalueHistoryLength: 20);

            var transformedData = pipelineAnomaly.Fit(data).Transform(data);

            var anomalies = context.Data.CreateEnumerable<SpikeAnomaly>(transformedData, reuseRowObject: false).ToList();

            var prices = data.GetColumn<float>("Close").ToArray();
            var dates = data.GetColumn<DateTime>("Date").ToArray();
            for (int i = 0; i < anomalies.Count; i++)
            {
                if (anomalies[i].Anomalies[0] == 1)
                {
                    Console.WriteLine($"{dates[i]}\t{prices[i]}");
                }
            }

            Console.WriteLine("***************************\tPREDICTION BEGINS\t***************************");

            var pipelinePrediction = context.Forecasting.ForecastBySsa(nameof(PricePrediction.Predictions),
                nameof(CurrencyModel.Close), windowSize: 5,
                seriesLength: 10, trainSize: 100, horizon: 2);
            var model = pipelinePrediction.Fit(data);
            var forecastContext = model.CreateTimeSeriesEngine<CurrencyModel, PricePrediction>(context);
            var forecasts = forecastContext.Predict(3);

            foreach (var item in forecasts.Predictions)
            {
                Console.WriteLine(item);
            }

            Console.ReadLine();
        }
    }
}