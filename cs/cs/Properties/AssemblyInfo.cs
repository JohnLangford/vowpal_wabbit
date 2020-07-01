using System.Runtime.CompilerServices;

//all the rest assembly properties are migrated inside .csproj file, these are not supported by msbuild yet
[assembly: System.Runtime.InteropServices.Guid("6a577997-af00-4ca0-8453-fdc8bbdf2a57")]
[assembly: System.Reflection.AssemblyTrademark("")]
[assembly: System.Runtime.InteropServices.ComVisible(false)]
[assembly: System.CLSCompliant(false)]

// make internals available to unit test
[assembly: InternalsVisibleTo("cs_unittest,PublicKey=" +
                              "0024000004800000940000000602000000240000525341310004000001000100515aa9bda65291" +
                              "811af92b381378bd271aff3a9e177bac69ff0e85874952fd82c0fbcb53f4e968181d07418481ee" +
                              "2be97522d44c324aa5c683dafaa449fe66ddc65e1d9b3c0600c8820bd2be6401c6888ea88864ef" +
                              "0b6ae5bfbf450aa1f548568d638913d82954195947e394c225cca2cd2f8132d525c2fdc0c57835" +
                              "b87200aa")]