using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using VW;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace cs_unittest
{
    [TestClass]
    public abstract class TestBase : IDisposable
    {
        public TestBase()
        {
            this.Init();
        }

        [TestInitialize]
        public void Init()
        {
            var basePath = Path.GetDirectoryName(typeof(TestBase).Assembly.Location);
            Environment.CurrentDirectory = basePath + @"\..\..\..\..\..\..\test";

            if (!Directory.Exists("models"))
            {
                Directory.CreateDirectory("models");
            }
        }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists("models"))
                {
                    Directory.Delete("models", true);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to delete model directory: "+ex.Message);
            }
        }
    }
}
