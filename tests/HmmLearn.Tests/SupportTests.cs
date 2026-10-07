using System.Text.Json;
using HmmLearn.Tests.Support;

namespace HmmLearn.Tests;

/// <summary>Checks the test support itself (golden loader and tolerance), independently of any model.</summary>
public class SupportTests
{
    [Fact]
    public void Repository_root_holds_the_solution_and_the_golden_folder()
    {
        Assert.True(File.Exists(Path.Combine(Golden.RepositoryRoot, "HmmLearn.sln")));
        Assert.True(Directory.Exists(Path.Combine(Golden.RepositoryRoot, "oracle", "golden")));
    }

    [Fact]
    public void Every_golden_file_parses()
    {
        var files = Directory.GetFiles(Path.Combine(Golden.RepositoryRoot, "oracle", "golden"), "*.json");
        Assert.NotEmpty(files);
        foreach (var file in files)
            Assert.NotEqual(JsonValueKind.Undefined, Golden.Load(Path.GetFileNameWithoutExtension(file)).ValueKind);
    }

    [Fact]
    public void Readers_build_row_major_arrays_and_decode_non_finite_numbers()
    {
        using var document = JsonDocument.Parse("""{"m": [[1, 2, 3], [4, 5, 6]], "t": [[[1, 2], [3, 4]]], "v": [1, "NaN", "-Infinity", "Infinity"]}""");
        var root = document.RootElement;
        var m = root.GetProperty("m").AsMatrix();
        Assert.Equal(2, m.GetLength(0));
        Assert.Equal(3, m.GetLength(1));
        Assert.Equal(4.0, m[1, 0]);
        var t = root.GetProperty("t").AsTensor3();
        Assert.Equal(3.0, t[0, 1, 0]);
        var v = root.GetProperty("v").AsVector();
        Assert.True(double.IsNaN(v[1]));
        Assert.True(double.IsNegativeInfinity(v[2]));
        Assert.True(double.IsPositiveInfinity(v[3]));
    }

    [Fact]
    public void Close_follows_numpy_allclose_semantics()
    {
        Close.Equal(1.0, 1.0 + 1e-9);
        Close.Equal(double.NaN, double.NaN);
        Close.Equal(double.NegativeInfinity, double.NegativeInfinity);
        Close.Equal(new double[,] { { 1, 2 }, { 3, 4 } }, new double[,] { { 1, 2 }, { 3, 4 + 1e-9 } });
        Assert.ThrowsAny<Exception>(() => Close.Equal(1.0, 1.001));
        Assert.ThrowsAny<Exception>(() => Close.Equal(double.NegativeInfinity, double.PositiveInfinity));
        Assert.ThrowsAny<Exception>(() => Close.Equal(new double[] { 1, 2 }, new double[] { 1, 2, 3 }));
        Assert.ThrowsAny<Exception>(() => Close.Equal(new double[,] { { 1 } }, new double[] { 1 }));
    }
}
