using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using cs_unittest;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using VW;
using VW.Labels;
using VW.Serializer.Attributes;
using System.Threading;
using VW.Serializer;
using cs_testcommon;

namespace cs_unittest
{
    [TestClass]
    public class Test1and2Class : TestBase
    {
        [TestMethod]
        [TestCategory("Vowpal Wabbit/Command line through marshalling")]
        public void Test1and2()
        {
            var references = File.ReadAllLines(@"pred-sets\ref\0001.predict").Select(l => float.Parse(l, CultureInfo.InvariantCulture)).ToArray();

            var input = new List<Test1>();

            using (var vwStr = new VowpalWabbit(" -k -c test1and2.str --passes 8 -l 20 --power_t 1 --initial_t 128000  --ngram 3 --skips 1 --invariant --holdout_off"))
            using (var vw = new VowpalWabbit<Test1>(new VowpalWabbitSettings(" -k -c test1and2 --passes 8 -l 20 --power_t 1 --initial_t 128000  --ngram 3 --skips 1 --invariant --holdout_off")
                { EnableExampleCaching = false }))
            using (var vwValidate = new VowpalWabbitExampleValidator<Test1>("-l 20 --power_t 1 --initial_t 128000  --ngram 3 --skips 1 --invariant --holdout_off"))
            {
                var lineNr = 0;
                VWTestHelper.ParseInput(
                    File.OpenRead(@"train-sets\0001.dat"),
                    new MyListener(data =>
                    {
                        input.Add(data);

                        vwValidate.Validate(data.Line, data, data.Label);

                        var expected = vwStr.Learn(data.Line, VowpalWabbitPredictionType.Dynamic);
                        Assert.IsInstanceOfType(expected, typeof(float));
                        var actual = vw.Learn(data, data.Label, VowpalWabbitPredictionType.Scalar);

                        Assert.AreEqual((float)expected, actual, 1e-6, "Learn output differs on line: " + lineNr);

                        lineNr++;
                    }));

                vwStr.RunMultiPass();
                vw.Native.RunMultiPass();

                vwStr.SaveModel("models/str0001.model");
                vw.Native.SaveModel("models/0001.model");

                VWTestHelper.AssertEqual(@"train-sets\ref\0001.stderr", vwStr.PerformanceStatistics);
                VWTestHelper.AssertEqual(@"train-sets\ref\0001.stderr", vw.Native.PerformanceStatistics);
            }

            Assert.AreEqual(input.Count, references.Length);

            using (var vwModel = new VowpalWabbitModel(new VowpalWabbitSettings("-k -t --invariant") { ModelStream = File.OpenRead("models/0001.model") }))
            using (var vwInMemoryShared1 = new VowpalWabbit(new VowpalWabbitSettings { Model = vwModel }))
            using (var vwInMemoryShared2 = new VowpalWabbit<Test1>(new VowpalWabbitSettings { Model = vwModel }))
            using (var vwInMemory = new VowpalWabbit(new VowpalWabbitSettings("-k -t --invariant") { ModelStream = File.OpenRead("models/0001.model") }))
            using (var vwStr = new VowpalWabbit("-k -t -i models/str0001.model --invariant"))
            using (var vwNative = new VowpalWabbit("-k -t -i models/0001.model --invariant"))
            using (var vw = new VowpalWabbit<Test1>("-k -t -i models/0001.model --invariant"))
            using (var vwModel2 = new VowpalWabbitModel("-k -t --invariant -i models/0001.model"))
            using (var vwInMemoryShared3 = new VowpalWabbit<Test1>(new VowpalWabbitSettings { Model = vwModel2 }))
            {
                for (var i = 0; i < input.Count; i++)
                {
                    var actualStr = vwStr.Predict(input[i].Line, VowpalWabbitPredictionType.Scalar);
                    var actualNative = vwNative.Predict(input[i].Line, VowpalWabbitPredictionType.Scalar);
                    var actualInMemory = vwInMemory.Predict(input[i].Line, VowpalWabbitPredictionType.Scalar);

                    var actual = vw.Predict(input[i], VowpalWabbitPredictionType.Scalar, input[i].Label);
                    var actualShared1 = vwInMemoryShared1.Predict(input[i].Line, VowpalWabbitPredictionType.Scalar);
                    var actualShared2 = vwInMemoryShared2.Predict(input[i], VowpalWabbitPredictionType.Scalar, input[i].Label);
                    var actualShared3 = vwInMemoryShared3.Predict(input[i], VowpalWabbitPredictionType.Scalar, input[i].Label);

                    Assert.AreEqual(references[i], actualStr, 1e-5);
                    Assert.AreEqual(references[i], actualNative, 1e-5);
                    Assert.AreEqual(references[i], actualInMemory, 1e-5);
                    Assert.AreEqual(references[i], actual, 1e-5);
                    Assert.AreEqual(references[i], actualShared1, 1e-5);
                    Assert.AreEqual(references[i], actualShared2, 1e-5);
                    Assert.AreEqual(references[i], actualShared3, 1e-5);
                }

                // due to shared usage the counters don't match up
                //VWTestHelper.AssertEqual(@"test-sets\ref\0001.stderr", vwInMemoryShared2.Native.PerformanceStatistics);
                //VWTestHelper.AssertEqual(@"test-sets\ref\0001.stderr", vwInMemoryShared1.PerformanceStatistics);

                VWTestHelper.AssertEqual(@"test-sets\ref\0001.stderr", vwInMemory.PerformanceStatistics);
                VWTestHelper.AssertEqual(@"test-sets\ref\0001.stderr", vwStr.PerformanceStatistics);
                VWTestHelper.AssertEqual(@"test-sets\ref\0001.stderr", vw.Native.PerformanceStatistics);
            }
        }
    }

    // 1|features 13:.1 15:.2 const:25
    // 1|abc 13:.1 15:.2 co:25
    public class Test1
    {
        [Feature(FeatureGroup = 'f', Namespace = "eatures", Name = "const", Order = 2)]
        public float Constant { get; set; }

        [Feature(FeatureGroup = 'f', Namespace = "eatures", Order = 1)]
        public IList<KeyValuePair<string, float>> Features { get; set; }

        public string Line { get; set; }

        public ILabel Label { get; set;}
    }

    public class Rcv1CbEval
    {
        [Feature]
        public string[] Words { get; set; }
    }

    public class MyListener : VowpalWabbitBaseListener
    {
        private Test1 example;

        private Action<Test1> action;

        public MyListener(Action<Test1> action)
        {
            this.action = action;
        }

        public override void EnterExample(VowpalWabbitParser.ExampleContext context)
        {
            this.example = new Test1()
            {
                Features = new List<KeyValuePair<string, float>>()
            };
        }

        public override void ExitExample(VowpalWabbitParser.ExampleContext context)
        {
            this.example.Line = context.GetText();
            this.action(this.example);
        }

        public override void ExitNumber(VowpalWabbitParser.NumberContext context)
        {
            context.value = float.Parse(context.GetText(), CultureInfo.InvariantCulture);
        }

        public override void ExitLabel_simple(VowpalWabbitParser.Label_simpleContext context)
        {
            this.example.Label = new SimpleLabel()
            {
                Label = context.value.value
            };
        }

        public override void ExitFeatureSparse(VowpalWabbitParser.FeatureSparseContext context)
        {
            var index = context.index.Text;
            var x = context.x.value;

            if (index == "const")
            {
                this.example.Constant = x;
            }
            else
            {
                this.example.Features.Add(new KeyValuePair<string, float>(index, x));
            }
        }
    }
}
