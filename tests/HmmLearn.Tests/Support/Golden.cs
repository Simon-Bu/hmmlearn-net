using System.Text.Json;

namespace HmmLearn.Tests.Support;

/// <summary>
/// Reads the golden files written by oracle/gen_oracle.py (numbers computed by hmmlearn 0.3.3, inputs always explicit)
/// and compares numbers with the oracle tolerance. Test support only: no model logic lives here.
/// </summary>
public static class Golden
{
    /// <summary>The folder that holds HmmLearn.sln, found by walking up from the test assembly.</summary>
    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    /// <summary>The parsed content of oracle/golden/{name}.json (name without extension, e.g. "utils").</summary>
    public static JsonElement Load(string name)
    {
        var path = Path.Combine(RepositoryRoot, "oracle", "golden", name + ".json");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory != null && !File.Exists(Path.Combine(directory.FullName, "HmmLearn.sln")))
            directory = directory.Parent;
        return directory?.FullName ?? throw new DirectoryNotFoundException("HmmLearn.sln not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>Typed readers for golden JSON: numbers, vectors, matrices and tensors in row-major order, like numpy.</summary>
public static class GoldenJson
{
    /// <summary>A JSON number, or one of the strings "NaN", "Infinity", "-Infinity" (JSON has no literal for them).</summary>
    public static double AsDouble(this JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Number => e.GetDouble(),
        JsonValueKind.String => e.GetString() switch
        {
            "NaN" => double.NaN,
            "Infinity" => double.PositiveInfinity,
            "-Infinity" => double.NegativeInfinity,
            var s => throw new FormatException($"not a number: {s}"),
        },
        _ => throw new FormatException($"not a number: {e.ValueKind}"),
    };

    public static int AsInt(this JsonElement e) => e.GetInt32();

    public static string AsString(this JsonElement e) => e.GetString() ?? throw new FormatException("null string");

    public static bool AsBool(this JsonElement e) => e.GetBoolean();

    /// <summary>True when the object has this property and it is not JSON null.</summary>
    public static bool Has(this JsonElement e, string name) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null;

    /// <summary>The elements of a JSON array (the cases of a golden file, for instance).</summary>
    public static IEnumerable<JsonElement> Items(this JsonElement e) => e.EnumerateArray();

    public static double[] AsVector(this JsonElement e) => (double[])Build(e, 1);

    public static double[,] AsMatrix(this JsonElement e) => (double[,])Build(e, 2);

    public static double[,,] AsTensor3(this JsonElement e) => (double[,,])Build(e, 3);

    public static double[,,,] AsTensor4(this JsonElement e) => (double[,,,])Build(e, 4);

    public static int[] AsInts(this JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    /// <summary>An array of arrays of integers (for instance the symbols of a categorical model, shape (n, 1)).</summary>
    public static int[,] AsIntMatrix(this JsonElement e)
    {
        var matrix = e.AsMatrix();
        var result = new int[matrix.GetLength(0), matrix.GetLength(1)];
        for (var i = 0; i < result.GetLength(0); i++)
            for (var j = 0; j < result.GetLength(1); j++)
                result[i, j] = checked((int)matrix[i, j]);
        return result;
    }

    private static Array Build(JsonElement e, int rank)
    {
        var flat = new List<double>();
        var shape = new int[rank];
        var seen = new bool[rank];
        Flatten(e, 0, rank, flat, shape, seen);
        var array = Array.CreateInstance(typeof(double), shape);
        Buffer.BlockCopy(flat.ToArray(), 0, array, 0, flat.Count * sizeof(double));
        return array;
    }

    private static void Flatten(JsonElement e, int depth, int rank, List<double> flat, int[] shape, bool[] seen)
    {
        if (depth == rank)
        {
            flat.Add(e.AsDouble());
            return;
        }
        if (e.ValueKind != JsonValueKind.Array) throw new FormatException($"expected an array at depth {depth} of {rank}");
        var length = e.GetArrayLength();
        if (!seen[depth]) { shape[depth] = length; seen[depth] = true; }
        else if (shape[depth] != length) throw new FormatException($"ragged array at depth {depth}: {shape[depth]} vs {length}");
        foreach (var item in e.EnumerateArray()) Flatten(item, depth + 1, rank, flat, shape, seen);
    }
}

/// <summary>
/// Number comparison with the oracle tolerance: |actual - expected| &lt;= atol + rtol * |expected| (numpy.allclose),
/// rtol = 1e-6 and atol = 1e-12 by default. NaN equals NaN; infinities must be equal.
/// </summary>
public static class Close
{
    public const double RelativeTolerance = 1e-6;
    public const double AbsoluteTolerance = 1e-12;

    public static bool IsClose(double expected, double actual, double rtol = RelativeTolerance, double atol = AbsoluteTolerance)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual)) return double.IsNaN(expected) && double.IsNaN(actual);
        if (double.IsInfinity(expected) || double.IsInfinity(actual)) return expected.Equals(actual);
        return Math.Abs(actual - expected) <= atol + rtol * Math.Abs(expected);
    }

    public static void Equal(double expected, double actual, string what = "value",
        double rtol = RelativeTolerance, double atol = AbsoluteTolerance)
    {
        if (!IsClose(expected, actual, rtol, atol))
            Assert.Fail($"{what}: expected {expected:R}, got {actual:R} (rtol {rtol}, atol {atol})");
    }

    /// <summary>Same rank, same shape, and every element close (works for double[], double[,], double[,,], double[,,,]).</summary>
    public static void Equal(Array expected, Array actual, string what = "array",
        double rtol = RelativeTolerance, double atol = AbsoluteTolerance)
    {
        if (expected.Rank != actual.Rank) Assert.Fail($"{what}: expected rank {expected.Rank}, got {actual.Rank}");
        for (var d = 0; d < expected.Rank; d++)
            if (expected.GetLength(d) != actual.GetLength(d))
                Assert.Fail($"{what}: dimension {d} has length {actual.GetLength(d)}, expected {expected.GetLength(d)}");
        var e = expected.Cast<double>().ToArray();
        var a = actual.Cast<double>().ToArray();
        for (var i = 0; i < e.Length; i++)
            if (!IsClose(e[i], a[i], rtol, atol))
                Assert.Fail($"{what}{Coordinates(expected, i)}: expected {e[i]:R}, got {a[i]:R} (rtol {rtol}, atol {atol})");
    }

    private static string Coordinates(Array array, int flatIndex)
    {
        var indexes = new int[array.Rank];
        for (var d = array.Rank - 1; d >= 0; d--)
        {
            indexes[d] = flatIndex % array.GetLength(d);
            flatIndex /= array.GetLength(d);
        }
        return "[" + string.Join(", ", indexes) + "]";
    }
}
