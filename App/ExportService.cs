using MiniChat.Storage;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>
/// /export : un modèle sauvegardé (model, model-pretrain, ou un nom de /save) en exports/&lt;nom&gt;.nelmodel,
/// pour /import dans Nel Code (voir NELMODEL-FORMAT.md).
/// </summary>
public sealed class ExportService
{
    private readonly ModelCatalog _models;
    private readonly string _exportsFolder;

    public ExportService(ModelCatalog models, string exportsFolder)
    {
        _models = models;
        _exportsFolder = exportsFolder;
    }

    /// <summary>Écrit le .nelmodel puis relit le fichier pour le vérifier. Un export du même nom est remplacé.</summary>
    public ExportResult Export(string name, string source, string? description)
    {
        if (!ModelCatalog.IsValidName(name))
            throw new UserException($"Nom invalide : « {name} ». Lettres, chiffres, - _ . (64 caractères au plus).");
        var store = _models.Require(source);

        var config = store.ReadConfig();
        var training = store.ReadInfo();
        string path = Path.Combine(_exportsFolder, name + NelModelFile.Extension);
        bool replaced = File.Exists(path);

        // Le modèle est chargé sur le CPU (la mémoire GPU reste au chat) : pour le vrai nombre de paramètres,
        // model.safetensors et reference.json (calculé en float32, sans TF32). Le chargement vérifie aussi que le
        // modèle a ses tokens spéciaux, sans lesquels Nel Code le refuse.
        string generated = Path.Combine(Path.GetTempPath(), "minichat-export-" + Guid.NewGuid().ToString("N"));
        NelModelInfo info;
        try
        {
            var (model, tokenizer) = store.Load(CPU);
            using (model)
            {
                info = new NelModelInfo(NelModelFile.FormatVersion, name, DateTimeOffset.Now.ToString("yyyy-MM-dd'T'HH:mm:sszzz"),
                    model.ParameterCount, config.Architecture, training?.ValLoss, training?.Step, description);
                Directory.CreateDirectory(generated);
                NelModelFile.WriteGenerated(model, tokenizer, generated);
            }

            NelModelFile.Write(store, generated, info, path);
            NelModelFile.Verify(path, store, info.Parameters!.Value);
        }
        catch (InvalidDataException e)
        {
            File.Delete(path);
            throw new UserException($"Export raté, fichier supprimé : {e.Message}");
        }
        finally
        {
            if (Directory.Exists(generated))
                Directory.Delete(generated, recursive: true);
        }
        return new ExportResult(path, _models.NameOf(source), info, new FileInfo(path).Length, replaced);
    }
}

/// <summary>Un .nelmodel écrit et vérifié par /export. Replaced = un export du même nom existait.</summary>
public sealed record ExportResult(string Path, string Source, NelModelInfo Info, long Bytes, bool Replaced);
