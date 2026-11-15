using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using MiniChat.Tokenization;

namespace MiniChat.Data;

/// <summary>
/// Les tokens d'un fichier de données gardés sur le disque, pour ne pas redécouper texte.txt (~1 min)
/// à chaque phase et à chaque lancement. Un fichier par nom : cache/&lt;nom&gt;-&lt;clé&gt;.bin, avec
///   en-tête : "MCTK", FormatVersion, empreinte SHA-256 des fichiers sources et de settings, empreinte du tokenizer,
///             onlyReplies, nombre de caractères, nombre de tokens (long ; int dans la version 1, encore relue)
///   ids     : int16, 2 octets par token
///   masque  : 1 octet par token (seulement pour les dialogues, onlyReplies)
/// Les tokens sont écrits et relus morceau par morceau (ChunkedArray) : un texte peut dépasser 2,1 milliards de
/// tokens, et le fichier ne dépend pas de la taille des morceaux.
/// Le cache n'est utilisé que si les deux empreintes correspondent : une source modifiée ou un autre tokenizer
/// (nouveau /train) = on redécoupe, et l'ancien fichier de ce nom est remplacé.
/// FormatVersion : à augmenter si le découpage change (BpeTokenizer, ChatFormat, TextDataset.Tokenize...).
/// </summary>
public static class TokenCache
{
    private const int FormatVersion = 2;
    private static readonly byte[] Magic = "MCTK"u8.ToArray();

    /// <summary>
    /// Les tokens tirés des fichiers sources : depuis le cache s'il correspond, sinon tokenize(), puis écrits dans le cache.
    /// settings : ce qui change le résultat sans être dans les fichiers (ex. le nombre de copies d'identite.txt).
    /// FromCache = vrai si rien n'a été redécoupé. chunkSize : taille des morceaux des tokens relus.
    /// </summary>
    public static (TokenizedText Text, bool FromCache) Load(string folder, string name, IReadOnlyList<string> sources,
        string settings, BpeTokenizer tokenizer, bool onlyReplies, Func<TokenizedText> tokenize,
        int chunkSize = ChunkedArray<short>.DefaultChunkSize)
    {
        byte[] source = SHA256.HashData([.. sources.SelectMany(HashFile), .. Encoding.UTF8.GetBytes(settings)]);
        byte[] tokens = Fingerprint(tokenizer);
        string key = Convert.ToHexString(SHA256.HashData([.. source, .. tokens, (byte)(onlyReplies ? 1 : 0)]))[..16];
        string path = Path.Combine(folder, $"{name}-{key}.bin");

        if (File.Exists(path) && TryRead(path, source, tokens, onlyReplies, chunkSize) is { } cached)
            return (cached, true);

        var text = tokenize();
        Write(folder, name, path, source, tokens, onlyReplies, text);
        return (text, false);
    }

    /// <summary>Empreinte du tokenizer : ses caractères, ses fusions et ses tokens spéciaux.</summary>
    private static byte[] Fingerprint(BpeTokenizer tokenizer)
    {
        var text = new StringBuilder(tokenizer.Alphabet).Append('\0');
        foreach (var (left, right) in tokenizer.Merges)
            text.Append(left).Append(',').Append(right).Append(';');
        text.Append('\0').AppendJoin('\0', tokenizer.SpecialTokens);
        return SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString()));
    }

    // ~1 s pour 1,4 Go.
    private static byte[] HashFile(string path)
    {
        using var file = File.OpenRead(path);
        return SHA256.HashData(file);
    }

    private static TokenizedText? TryRead(string path, byte[] source, byte[] tokens, bool onlyReplies, int chunkSize)
    {
        try
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (!reader.ReadBytes(Magic.Length).SequenceEqual(Magic))
                return null;
            int version = reader.ReadInt32();
            if (version is not (1 or FormatVersion)
                || !reader.ReadBytes(source.Length).SequenceEqual(source) || !reader.ReadBytes(tokens.Length).SequenceEqual(tokens)
                || reader.ReadBoolean() != onlyReplies)
                return null;

            long characters = reader.ReadInt64();
            long count = version == 1 ? reader.ReadInt32() : reader.ReadInt64();   // version 1 : nombre en int
            var ids = ReadChunks<short>(reader.BaseStream, count, chunkSize);
            var mask = onlyReplies ? ReadChunks<byte>(reader.BaseStream, count, chunkSize) : null;
            return new TokenizedText(ids, mask, characters);
        }
        catch (Exception e) when (e is IOException or EndOfStreamException)
        {
            return null;   // fichier coupé (arrêt pendant l'écriture) : on redécoupe
        }
    }

    // count éléments lus directement dans des morceaux de chunkSize : jamais plus que leur place en mémoire.
    private static ChunkedArray<T> ReadChunks<T>(Stream stream, long count, int chunkSize) where T : unmanaged
    {
        var builder = new ChunkedArray<T>.Builder(chunkSize);
        for (long done = 0; done < count; done += chunkSize)
        {
            var chunk = new T[(int)Math.Min(chunkSize, count - done)];
            stream.ReadExactly(MemoryMarshal.AsBytes(chunk.AsSpan()));
            builder.AddChunk(chunk);
        }
        return builder.Build();
    }

    // Écrit dans un .tmp puis renomme : une coupure pendant l'écriture ne laisse pas de cache à moitié écrit.
    private static void Write(string folder, string name, string path, byte[] source, byte[] tokens, bool onlyReplies,
        TokenizedText text)
    {
        Directory.CreateDirectory(folder);
        foreach (string old in Directory.EnumerateFiles(folder, $"{name}-*.bin"))
            File.Delete(old);

        using (var writer = new BinaryWriter(File.Create(path + ".tmp")))
        {
            writer.Write(Magic);
            writer.Write(FormatVersion);
            writer.Write(source);
            writer.Write(tokens);
            writer.Write(onlyReplies);
            writer.Write(text.Characters);
            writer.Write(text.Ids.Length);
            // Morceau par morceau : les octets d'un morceau tiennent dans un Span (2 Go au plus), pas ceux du texte entier.
            foreach (var chunk in text.Ids.Chunks)
                writer.Write(MemoryMarshal.AsBytes(chunk.AsSpan()));
            if (onlyReplies)
                foreach (var chunk in text.Mask!.Chunks)
                    writer.Write(chunk);
        }
        File.Move(path + ".tmp", path, overwrite: true);
    }
}
