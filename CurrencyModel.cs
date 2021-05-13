using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnomalyDetectionSample
{
    public class CurrencyModel
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Open { get; set; }

        [LoadColumn(2)]
        public float High { get; set; }

        [LoadColumn(3)]
        public float Low { get; set; }

        [LoadColumn(4)]
        public float Close { get; set; }

        [LoadColumn(5)]
        public float Volume { get; set; }

        [LoadColumn(6)]
        public float MarketCap { get; set; }
    }

    public class SpikeAnomaly
    {
        [VectorType(2)]
        public double[] Anomalies { get; set; }
    }

    public class PricePrediction
    {
        [VectorType(2)]
        public float[] Predictions { get; set; }
    }
}