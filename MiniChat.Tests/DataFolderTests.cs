using System.Reflection;
using MiniChat.App;
using Xunit;

namespace MiniChat.Tests;

public sealed class DataFolderTests
{
    [Fact]
    public void ExecutableKnowsTheProjectDataFolder()
    {
        // MiniChat lit Data/ du projet (la compilation ne le copie plus dans bin/) : son chemin est noté dans l'assembly.
        string? folder = typeof(Workbench).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == "ProjectDataFolder")?.Value;

        Assert.NotNull(folder);
        Assert.Equal(Path.GetFullPath(Path.GetDirectoryName(TestData.ProjectDataFile("eval.json"))!), Path.GetFullPath(folder));
        Assert.True(File.Exists(Path.Combine(folder, "eval.json")));
    }

    [Fact]
    public void DataIsNotCopiedNextToTheExecutable() =>
        // MiniChat.dll est à côté des tests : ses données, elles, ne doivent pas y être.
        Assert.False(Directory.Exists(Path.Combine(AppContext.BaseDirectory, "Data")));
}
