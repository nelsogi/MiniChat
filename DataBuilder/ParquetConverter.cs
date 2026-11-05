using System.Text;
using Parquet;
using Parquet.Schema;

namespace DataBuilder;

/// <summary>
/// Extrait le texte d'une colonne de fichiers .parquet (Wikipedia, FineWeb... sur Hugging Face)
/// pour le pré-entraînement. Chaque valeur de la colonne est un document.
///
/// Le nettoyage est celui de Gutenberg (GutenbergCleaner.CleanParagraph), mais ici une ligne = un paragraphe :
/// ces textes ne sont pas coupés à ~70 caractères comme les livres, et beaucoup séparent leurs paragraphes
/// par un seul saut de ligne. Résultat : un paragraphe par ligne.
///
/// Après Convert(), les compteurs (Kept, Empty...) disent ce qui a été gardé.
/// </summary>
public sealed class ParquetConverter
{
    private readonly string _column;
    private readonly long? _maxBytes;

    /// <summary>Documents écrits dans le fichier.</summary>
    public int Kept { get; private set; }

    /// <summary>Documents vides (ou null) après nettoyage.</summary>
    public int Empty { get; private set; }

    /// <summary>Vrai si la lecture s'est arrêtée à la taille maximale, avant la fin des fichiers.</summary>
    public bool ReachedMaxSize { get; private set; }

    /// <param name="column">Nom de la colonne de texte.</param>
    /// <param name="maxBytes">Taille max de la sortie en octets UTF-8 (null = pas de limite).</param>
    public ParquetConverter(string column, long? maxBytes)
    {
        _column = column;
        _maxBytes = maxBytes;
    }

    /// <returns>Le texte nettoyé. Un document qui ferait dépasser la taille max n'est pas gardé.</returns>
    public async Task<string> ConvertAsync(IEnumerable<string> inputPaths)
    {
        var output = new StringBuilder();
        long bytes = 0;

        foreach (string path in inputPaths)
        {
            await using var reader = await ParquetReader.CreateAsync(path);
            DataField field = FindColumn(reader.Schema, path);
            int keptBefore = Kept;

            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                using var rowGroup = reader.OpenRowGroupReader(i);
                var documents = new string?[rowGroup.RowCount];
                await rowGroup.ReadAsync(field, documents.AsMemory());

                foreach (string? document in documents)
                {
                    string cleaned = Clean(document);
                    if (cleaned.Length == 0)
                    {
                        Empty++;
                        continue;
                    }

                    long size = Encoding.UTF8.GetByteCount(cleaned);
                    if (bytes + size > _maxBytes)
                    {
                        ReachedMaxSize = true;
                        Console.WriteLine($"  {Path.GetFileName(path)} : {Kept - keptBefore:N0} documents (taille max atteinte).");
                        return output.ToString();
                    }

                    output.Append(cleaned);
                    bytes += size;
                    Kept++;
                }
            }

            Console.WriteLine($"  {Path.GetFileName(path)} : {Kept - keptBefore:N0} documents.");
        }

        return output.ToString();
    }

    /// <summary>Un paragraphe nettoyé par ligne, "" si rien ne reste.</summary>
    private static string Clean(string? document)
    {
        if (string.IsNullOrWhiteSpace(document)) return "";

        var paragraphs = document.Replace("\r\n", "\n").Replace('\r', '\n')
            .Split('\n')
            .Select(GutenbergCleaner.CleanParagraph)
            .Where(p => p.Length > 0);
        string text = string.Join('\n', paragraphs);
        return text.Length > 0 ? text + '\n' : "";
    }

    private DataField FindColumn(ParquetSchema schema, string path)
    {
        var fields = schema.GetDataFields();
        // Parquet.Net 6 annonce les colonnes de texte en ReadOnlyMemory<char> (string dans les versions d'avant).
        return fields.FirstOrDefault(f => f.Name == _column && IsText(f))
            ?? throw new InvalidDataException(
                $"{Path.GetFileName(path)} : pas de colonne de texte '{_column}'. "
                + $"Colonnes : {string.Join(", ", fields.Select(f => $"{f.Name} ({(IsText(f) ? "texte" : f.ClrType.Name)})"))}. "
                + "Choisis-en une avec --column.");
    }

    // Parquet.Net 6 annonce les colonnes de texte en ReadOnlyMemory<char> (string dans les versions d'avant).
    private static bool IsText(DataField field) =>
        field.ClrType == typeof(ReadOnlyMemory<char>) || field.ClrType == typeof(string);
}
