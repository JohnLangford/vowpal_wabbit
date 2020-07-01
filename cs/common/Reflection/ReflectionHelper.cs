// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ReflectionHelper.cs">
//   Copyright (c) by respective owners including Yahoo!, Microsoft, and
//   individual contributors. All rights reserved.  Released under a BSD
//   license as described in the file LICENSE.
// </copyright>
// --------------------------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Reflection.Emit;

namespace VW.Reflection
{
    /// <summary>
    /// Reflection helper to find methods on visitors.
    /// </summary>
    public static class ReflectionHelper
    {
        /// <summary>
        /// Compiles the supplied <paramref name="sourceExpression"/> to a callable function.
        /// </summary>
        /// <param name="sourceExpression">The source expression to be compiled.</param>
        /// <returns>A callable function.</returns>
        /// <remarks>Can't constraint on Func (or would have to have 11 overloads) nor is it possible to constaint on delegate.</remarks>
#if NET452
        public static System.Delegate CompileToFunc<T>(this Expression<T> sourceExpression)
        {
            // inspect T to be Func<...>
            var funcType = typeof(T);

            if (!funcType.Name.StartsWith("Func`"))
                throw new ArgumentException("T must be one of the System.Func<...> type.");

            var genericArguments = funcType.GetGenericArguments();
            var returnType = genericArguments.Last();
            var paramTypes = genericArguments.Take(genericArguments.Length - 1);

            // sign serializer so we can get access to internal members.
            var asmName = new AssemblyName("VowpalWabbitSerializer");
            StrongNameKeyPair kp;
            using (var stream = typeof(ReflectionHelper).Assembly.GetManifestResourceStream("VW.vw_key.snk"))
            using (var memStream = new MemoryStream())
            {
                stream.CopyTo(memStream, 1024);

                kp = new StrongNameKeyPair(memStream.ToArray());
            }
            asmName.KeyPair = kp;

            var dynAsm = AppDomain.CurrentDomain.DefineDynamicAssembly(asmName, AssemblyBuilderAccess.RunAndSave);

            // Create a dynamic module and type
            //#if !DEBUG
            //var moduleBuilder = dynAsm.DefineDynamicModule("VowpalWabbitSerializerModule", asmName.Name + ".dll", true);
            //#else
            var moduleBuilder = dynAsm.DefineDynamicModule("VowpalWabbitSerializerModule");

            var typeBuilder = moduleBuilder.DefineType("VowpalWabbitSerializer" + Guid.NewGuid().ToString().Replace('-', '_'));

            // Create our method builder for this type builder
            const string methodName = "Method";
            var methodBuilder = typeBuilder.DefineMethod(
                methodName,
                MethodAttributes.Public | MethodAttributes.Static,
                returnType,
                paramTypes.ToArray());

            // compared to Compile this looks rather ugly, but there is a feature-bug
            // that adds a security check to every call of the Serialize method
            //#if !DEBUG
            //var debugInfoGenerator = DebugInfoGenerator.CreatePdbGenerator();
            //visit.CompileToMethod(methodBuilder, debugInfoGenerator);
            //#else
            sourceExpression.CompileToMethod(methodBuilder);
            //#endif

            var dynType = typeBuilder.CreateType();

            // for debugging only
            // dynAsm.Save(@"my.dll");

            return Delegate.CreateDelegate(typeof(T), dynType.GetMethod(methodName));
        }
#elif NETCOREAPP3_1
        public static System.Delegate CompileToFunc<T, TR>(this Expression<Func<T, TR>> sourceExpression)
        {
            return sourceExpression.Compile();
        }
#endif

        /// <summary>
        /// TODO: replace me with Roslyn once it's released and just generate string code. This way the overload resolution is properly done.
        /// </summary>
        /// <remarks>This is a simple heuristic for overload resolution, not the full thing.</remarks>
        public static MethodInfo FindMethod(Type objectType, string name, params Type[] parameterTypes)
        {
            Contract.Requires(objectType != null);
            Contract.Requires(name != null);
            Contract.Requires(parameterTypes != null);

            // let's find the "best" match:
            // order by
            //  1. distance (0 = assignable, 1 = using generic) --> ascending
            //  2. # of interfaces implemented. the more the better (the more specific we are) --> descending
            //  3. # of open generics. the less the better (the more specific we are) --> ascending
            var methods = from m in objectType.GetMethods(BindingFlags.Instance | BindingFlags.Public)
                          where m.Name == name
                          let parameters = m.GetParameters()
                          where parameters.Length == parameterTypes.Length
                          let output = parameterTypes.Zip(parameters, (valueType, methodParameter) => Distance(methodParameter.ParameterType, valueType)).ToArray()
                          where output.All(o => o != null)
                          let distance = output.Sum(o => o.Distance)
                          let interfacesImplemented = output.Sum(o => o.InterfacesImplemented)
                          let generics = output.Sum(o => o.GenericTypes.Count)
                          orderby
                           distance,
                           generics,
                           interfacesImplemented descending
                          select new
                          {
                              Method = m,
                              Distance = distance,
                              InterfacesImplemented = interfacesImplemented,
                              GenericTypes = output.Select(o => o.GenericTypes)
                          };

            var bestCandidate = methods.FirstOrDefault();
            if (bestCandidate == null)
            {
                return null;
            }

            MethodInfo method = bestCandidate.Method;

            //Debug.WriteLine("Method Search");
            //foreach (var item in methods)
            //{
            //    Debug.WriteLine(string.Format("Distance={0} Interfaces={1} OpenGenerics={2} Method={3}",
            //        item.Distance,
            //        item.InterfacesImplemented,
            //        item.GenericTypes.Count(gt => gt.Count > 0),
            //        item.Method));
            //}

            if (method.IsGenericMethod)
            {
                var mergedGenericTypes = bestCandidate.GenericTypes.SelectMany(d => d).ToLookup(kvp => kvp.Key, kvp => kvp.Value);

                // consistency check
                foreach (var gt in mergedGenericTypes)
                {
                    var refElem = gt.First();
                    if (gt.Any(t => t != refElem))
                    {
                        throw new NotSupportedException("Inconsistent generic argument mapping: " + string.Join(",", gt));
                    }
                }

                // map generic arguments to actual argument
                var actualTypes = method.GetGenericArguments().Select(t => mergedGenericTypes[t].First()).ToArray();

                method = method.MakeGenericMethod(actualTypes);
                //Debug.WriteLine("\t specializing: " + method);
            }
            // Debug.WriteLine("Method: {0} for {1} {2}", method, name, string.Join(",", parameterTypes.Select(t => t.ToString())));

            return method;
        }

        internal static TypeMatch Distance(Type candidate, Type valueType)
        {
            if (candidate == valueType)
            {
                return new TypeMatch(0)
                {
                    InterfacesImplemented = candidate.GetInterfaces().Count()
                };
            }

            if (candidate.IsAssignableFrom(valueType))
            {
                return new TypeMatch(1)
                {
                    InterfacesImplemented = candidate.GetInterfaces().Count()
                };
            }

            if (candidate.IsGenericParameter && candidate.GetGenericParameterConstraints().All(c => c.IsAssignableFrom(valueType)))
            {
                return new TypeMatch(2, candidate, valueType)
                {
                    InterfacesImplemented = candidate.GetInterfaces().Count()
                };
            }

            if (candidate.IsGenericType)
            {
                // try to find a match that is assignable...
                //
                var genericCandidate = candidate.GetGenericTypeDefinition();

                var bestMatches =
                    from typeDistance in valueType.GetInterfaces().Select(it => new TypeDistance { Distance = 1, Type = it })
                                .Union(GetBaseTypes(valueType))
                    let type = typeDistance.Type
                    where type.IsGenericType && type.GetGenericTypeDefinition() == genericCandidate
                    let distances = candidate.GetGenericArguments().Zip(type.GetGenericArguments(), (a, b) => Distance(a, b)).ToList()
                    where distances.All(d => d != null)
                    let output = new TypeMatch(typeDistance.Distance, distances)
                    {
                        InterfacesImplemented = distances.Sum(d => d.InterfacesImplemented)
                                + (candidate.IsInterface ? candidate.GetInterfaces().Count() : 0)
                    }
                    orderby output.Distance, output.InterfacesImplemented descending, output.GenericTypes.Count
                    select output;

                return bestMatches.FirstOrDefault();
            }

            return null;
        }

        internal static IEnumerable<TypeDistance> GetBaseTypes(Type type, int depth = 0)
        {
            if (type == typeof(object) || type == null)
            {
                yield break;
            }

            yield return new TypeDistance { Type = type, Distance = depth };

            foreach (var item in GetBaseTypes(type.BaseType, depth + 1))
            {
                yield return item;
            }
        }

        /// <summary>
        /// Gets the member info in a sort of type safe manner - it's better than using strings, but some runtime errors are still possbile.
        /// </summary>
        public static MemberInfo GetInfo<T, TResult>(Expression<Func<T, TResult>> expression)
        {
            Contract.Requires(expression != null);

            return GetInfo(expression.Body);
        }

        /// <summary>
        /// Gets the member info in a sort of type safe manner - it's better than using strings, but some runtime errors are still possbile.
        /// </summary>
        public static MemberInfo GetInfo<T>(Expression<Action<T>> expression)
        {
            Contract.Requires(expression != null);

            return GetInfo(expression.Body);
        }

        /// <summary>
        /// Gets the member info in a sort of type safe manner - it's better than using strings, but some runtime errors are still possbile.
        /// </summary>
        public static MemberInfo GetInfo(Expression expression)
        {
            Contract.Requires(expression != null);

            var binaryExpression = expression as BinaryExpression;
            if (binaryExpression != null)
            {
                if (binaryExpression.Method != null)
                {
                    return binaryExpression.Method;
                }

                throw new NotSupportedException();
            }

            var methodExpression = expression as MemberExpression;
            if (methodExpression != null)
            {
                return methodExpression.Member;
            }

            var methodCallExpression = expression as MethodCallExpression;
            if (methodCallExpression != null)
            {
                return methodCallExpression.Method;
            }

            var newExpression = expression as NewExpression;
            if (newExpression != null)
            {
                return newExpression.Constructor;
            }

            var unaryExpression = expression as UnaryExpression;
            if (unaryExpression != null)
            {
                if (unaryExpression.Method != null)
                {
                    return unaryExpression.Method;
                }
            }

            throw new NotSupportedException();
        }
    }
}
