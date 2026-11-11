using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using MiniChat.Model;
using static TorchSharp.torch;

namespace MiniChat.Storage;

/// <summary>
/// Les poids au format safetensors (https://github.com/huggingface/safetensors), lisible sans TorchSharp
/// (Python, Rust, JavaScript...) : model.safetensors du .nelmodel, voir NELMODEL-FORMAT.md.
///   8 octets     : N, taille de l'en-tête (entier non signé, little-endian)
///   N octets     : l'en-tête JSON, complété d'espaces jusqu'à un multiple de 8
///                  {"__metadata__": {...}, "nom": {"dtype": "F32", "shape": [...], "data_offsets": [début, fin]}, ...}
///   le reste     : les données, float32 little-endian, chaque tenseur à la suite (ordre de ligne, comme en C) ;
///                  data_offsets compte depuis le début de cette partie.
/// Les noms sont ceux de state_dict() de TorchSharp, les mêmes que dans weights.dat.
/// </summary>
public static class SafeTensors
{
    /// <summary>Écrit tous les poids de model (float32), dans l'ordre de state_dict().</summary>
    public static void Write(MiniGpt model, string path, IReadOnlyDictionary<string, string> metadata)
    {
        var tensors = model.state_dict().Select(p => (p.Key, Tensor: p.Value.detach().cpu().to(ScalarType.Float32).contiguous())).ToList();
        try
        {
            var header = new JsonObject { ["__metadata__"] = new JsonObject(metadata.Select(m => KeyValuePair.Create(m.Key, (JsonNode?)m.Value))) };
            long offset = 0;
            foreach (var (name, tensor) in tensors)
            {
                long bytes = tensor.numel() * sizeof(float);
                header[name] = new JsonObject
                {
                    ["dtype"] = "F32",
                    ["shape"] = new JsonArray(tensor.shape.Select(d => (JsonNode?)d).ToArray()),
                    ["data_offsets"] = new JsonArray(offset, offset + bytes),
                };
                offset += bytes;
            }

            byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
            int padded = (json.Length + 7) / 8 * 8;

            using var file = File.Create(path);
            Span<byte> size = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(size, (ulong)padded);
            file.Write(size);
            file.Write(json);
            file.Write(Enumerable.Repeat((byte)' ', padded - json.Length).ToArray());
            foreach (var (_, tensor) in tensors)
                file.Write(MemoryMarshal.AsBytes(tensor.data<float>().ToArray().AsSpan()));   // little-endian : x86 et ARM
        }
        finally
        {
            foreach (var (_, tensor) in tensors)
                tensor.Dispose();
        }
    }

    /// <summary>
    /// Vérifie qu'un model.safetensors contient exactement les poids de model : mêmes noms, formes et valeurs
    /// (à l'octet près). Lance InvalidDataException au premier écart.
    /// </summary>
    public static void Verify(string path, MiniGpt model)
    {
        using var file = File.OpenRead(path);
        Span<byte> size = stackalloc byte[8];
        file.ReadExactly(size);
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(size);
        if (headerLength > 100_000_000 || headerLength % 8 != 0)
            throw new InvalidDataException($"model.safetensors : en-tête de {headerLength} octets, illisible.");

        var headerBytes = new byte[headerLength];
        file.ReadExactly(headerBytes);
        var header = JsonNode.Parse(headerBytes)?.AsObject() ?? throw new InvalidDataException("model.safetensors : en-tête illisible.");
        long dataStart = 8 + (long)headerLength;

        var weights = model.state_dict();
        var names = header.Select(h => h.Key).Where(k => k != "__metadata__").ToHashSet();
        if (!names.SetEquals(weights.Keys))
            throw new InvalidDataException($"model.safetensors : noms différents de ceux du modèle " +
                                           $"({string.Join(", ", names.Except(weights.Keys).Concat(weights.Keys.Except(names)).Take(5))}...).");

        foreach (var (name, weight) in weights)
        {
            var entry = header[name]!;
            long[] shape = entry["shape"]!.AsArray().Select(d => (long)d!).ToArray();
            long begin = (long)entry["data_offsets"]![0]!, end = (long)entry["data_offsets"]![1]!;
            if ((string?)entry["dtype"] != "F32" || !shape.SequenceEqual(weight.shape) || end - begin != weight.numel() * sizeof(float))
                throw new InvalidDataException($"model.safetensors : {name} n'a pas le type ou la forme du modèle.");

            var stored = new byte[end - begin];
            file.Position = dataStart + begin;
            file.ReadExactly(stored);
            using var expected = weight.detach().cpu().contiguous();
            if (!MemoryMarshal.AsBytes(expected.data<float>().ToArray().AsSpan()).SequenceEqual(stored))
                throw new InvalidDataException($"model.safetensors : les valeurs de {name} diffèrent de weights.dat.");
        }
    }
}
