using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using MiniChat.Tokenization;

namespace MiniChat.Data;

/// <summary>
/// Les tokens d'un fichier de données gardés sur le disque, pour ne pas redécouper texte.txt (~1 min)
/// à chaque phase et à chaque lancement. Un fichier par source : cache/&lt;nom&gt;-&lt;clé&gt;.bin, avec
///   en-tête : "MCTK", FormatVersion, empreinte SHA-256 du fichier source, empreinte du tokenizer,
///             onlyReplies, nombre de caractères, nombre de tokens
///   ids     : int16, 2 octets par token
///   masque  : 1 octet par token (seulement pour les dialogues, onlyReplies)
/// Le cache n'est utilisé que si les deux empreintes correspondent : source modifiée ou autre tokenizer
/// (nouveau /train) = on redécoupe, et l'ancien fichier de cette source est remplacé.
/// FormatVersion : à augmenter si le découpage change (BpeTokenizer, ChatFormat, TextDataset.Tokenize...).
/// </summary>
public static class TokenCache
{
    private const int FormatVersion = 1;
    private static readonly byte[] Magic = "MCTK"u8.ToArray();

    /// <summary>
    /// Les tokens de sourcePath : depuis le cache s'il correspond, sinon tokenize(), puis écrits dans le cache.
    /// FromCache = vrai si rien n'a été redécoupé.
    /// </summary>
    public static (TokenizedText Text, bool FromCache) Load(string folder, string sourcePath, BpeTokenizer tokenizer,
        bool onlyReplies, Func<TokenizedText> tokenize)
    {
        byte[] source = HashFile(sourcePath);
        byte[] tokens = Fingerprint(tokenizer);
        string name = Path.GetFileNameWithoutExtension(sourcePath);
        string key = Convert.ToHexString(SHA256.HashData([.. source, .. tokens, (byte)(onlyReplies ? 1 : 0)]))[..16];
        string path = Path.Combine(folder, $"{name}-{key}.bin");

        if (File.Exists(path) && TryRead(path, source, tokens, onlyReplies) is { } cached)
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

    private static TokenizedText? TryRead(string path, byte[] source, byte[] tokens, bool onlyReplies)
    {
        try
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (!reader.ReadBytes(Magic.Length).SequenceEqual(Magic) || reader.ReadInt32() != FormatVersion
                || !reader.ReadBytes(source.Length).SequenceEqual(source) || !reader.ReadBytes(tokens.Length).SequenceEqual(tokens)
                || reader.ReadBoolean() != onlyReplies)
                return null;

            long characters = reader.ReadInt64();
            var ids = new short[reader.ReadInt32()];
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(ids.AsSpan()));
            byte[]? mask = null;
            if (onlyReplies)
            {
                mask = new byte[ids.Length];
                reader.BaseStream.ReadExactly(mask);
            }
            return new TokenizedText(ids, mask, characters);
        }
        catch (Exception e) when (e is IOException or EndOfStreamException)
        {
            return null;   // fichier coupé (arrêt pendant l'écriture) : on redécoupe
        }
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
            writer.Write(MemoryMarshal.AsBytes(text.Ids.AsSpan()));
            if (onlyReplies)
                writer.Write(text.Mask!);
        }
        File.Move(path + ".tmp", path, overwrite: true);
    }
}
