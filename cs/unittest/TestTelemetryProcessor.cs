using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.DataContracts;
using Microsoft.ApplicationInsights.Extensibility;
using Microsoft.ApplicationInsights.Extensibility.Implementation;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace cs_unittest
{
    public class TestTelemetryProcessor: ITelemetryProcessor
    {
        private ITelemetryProcessor Next { get; set; }

        public ConcurrentBag<ExceptionTelemetry> Exceptions { get; private set; }

        // Link processors to each other in a chain.
        public TestTelemetryProcessor(ITelemetryProcessor next, ConcurrentBag<ExceptionTelemetry> exceptions)
        {
            this.Next = next;
            this.Exceptions = exceptions;
        }

        public void Process(ITelemetry item)
        {
            var tt = item as TraceTelemetry;
            if (tt != null)
            {
                Console.WriteLine($"Trace: {tt.Message}");
                foreach (var prop in tt.Properties)
                    Console.WriteLine($"\t{prop.Key}: {prop.Value}");
            }

            var et = item as ExceptionTelemetry;
            if (et != null)
            {
                this.Exceptions.Add(et);
                Console.WriteLine($"Exception: {et.Message}.  {et.Exception.StackTrace}");
            }

            this.Next.Process(item);
        }
    }
}
