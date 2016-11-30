# tensorflow-special-octo-spoon

## Writing TensorFlow Records

It is possible to write TFRecords using the `tf.python_io.TFRecordWriter`
class, which can be used in Python `with` blocks, and whose purpose is to write
records to a TFRecords file.

A `tf.python_io.TFRecordWriter` object is initialized with the filename of the
output file. We want to call the `tf.python_io.TFRecordWriter.write` method on
(TODO(brendan): refer to Features protobuf format)

### <a name="protobuf-heading"></a>Google Protobuf Format

#### Introduction

Google Protocol Buffers (a.k.a. protobuf) are a binary format of data storage
that can be used in various applications, e.g. in TensorFlow for storing
training and test datasets.

The advantages of using protobufs are in performance and data size: compared
with XML, protobufs are 3 to 10 times smaller, and parsing of protobufs is 20
to 100 times faster.

The main disadvantage of protobufs is that, since it is a binary format, it is
not human readable, and a `.proto` file is needed to describe the structure of
the data so that it can be de-serialized (i.e. converted from binary format
into the object structure of a programming language such as Python).

To read more about the motivation and history behind protobufs, see the
[Protocol Buffers Overview](https://developers.google.com/protocol-buffers/docs/overview).

#### `Features` Example

##### Syntax

As an example, take the following excerpt from
tensorflow/core/example/feature.proto.

<a name="feature-protobuf-code"></a>
```protobuf
// Containers to hold repeated fundamental values.
message BytesList {
  repeated bytes value = 1;
}
message FloatList {
  repeated float value = 1 [packed = true];
}
message Int64List {
  repeated int64 value = 1 [packed = true];
}

// Containers for non-sequential data.
message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};
```

Note that the `syntax = "proto3";` line in the `feature.proto` file indicates
that the proto3 (as opposed to proto2) version of the protobuf standard is
used.

`BytesList`, `FloatList`, `Int64List`, `Feature`, and `Features` are all
message types.

Each field in each message type definition has a unique numbered tag. For
example, the field `FloatList float_list` in message type `Feature` has a tag
of 2.

The `repeated` keyword indicates that a message may have zero or more of this
field, e.g. `BytesList` is made up of a sequence of zero or more `bytes`.

The `[packed = true]` syntax is a holdover from proto2, since `repeated` values
are using packed encoding by default in proto3. The meaning of packed encoding
can be found in the [protobuf encoding
guide](https://developers.google.com/protocol-buffers/docs/encoding#packed).

When an encoded message is parsed, missing fields are given [default
values](https://developers.google.com/protocol-buffers/docs/proto3#default).
For example, for numeric types such as `int64` the default value is zero.

Message types can be fields within other message types, for example
in `Feature`, `float_list` is a field of type `FloatList`.

A `oneof` field can be thought of as a `union` type in C, where each field in a
`oneof` block shares memory, and at most one field can be set at the same time.
Setting a `oneof` field will automatically clear all other members of the
`oneof` block, so if we have a `Feature` and set its `bytes_list` field, that
value will be deleted if then set its `float_list` field.

The `map` syntax is a shortcut for generating key, value pairs, where the key
type is any integral or string type and the value can be anything. The
`map<string, Feature> feature = 1;` line uses this syntax, where the key is a
`string` and the value is a `Feature`. The `map` syntax from our example line
is equivalent to the code block below.

```protobuf
message MapFieldEntry {
        string key = 1;
        Feature value = 2;
}

repeated MapFieldEntry feature = 1;
```

So in our example, the `map` usage is making use of the fact that message type
definitions can be nested.

Based on the `.proto` file, the [protocol buffer
compiler](https://developers.google.com/protocol-buffers/docs/proto3#generating)
for Python will generate a module with a static descriptor of each message
type. This generated module is then used with a _metaclass_ to create the
necessary Python data access class at runtime.
TODO(brendan): More concise summary/refer to tutorial

More detailed information about proto3 syntax can be found in the [proto3
language guide](https://developers.google.com/protocol-buffers/docs/proto3).


##### Varints

To understand protobuf encoding we must first quickly cover varints. Protobufs
use base 128 varints, where the value is little-endian (i.e. the least
significant byte comes first), each byte contains 7 bits of the value and the
eighth bit (most-significant bit) is set to 1 if there are more bytes to come,
and 0 if it is the last byte.

For example, the value 300 is `ac 02`, where we get 300 by dropping the high
bit from the first byte `ac` to get `2c`, and shifting the second (and last)
byte down by 1 since we dropped a bit in the first byte: `2c 01`. `2c 01` read
in little-endian is 0x12c, or 300.

##### Encoding

We will use the following code snippet from the `write_tfrecords` function from
`read_mnist_data.py` as an example to explain protobuf encoding.

```python
image_raw = dataset.images[example_index].tostring()
feature = {
        'height': int64_feature(dataset.images.shape[1]),
        'width': int64_feature(dataset.images.shape[2]),
        'label': int64_feature(int(dataset.labels[example_index])),
        'image_raw': bytes_feature(image_raw)
}
next_features = tf.train.Features(feature = feature)
example = tf.train.Example(features = next_features)
writer.write(example.SerializeToString())
```

The `Features` protobuf is as given [above](#feature-protobuf-code) in the
syntax section, and the `Example` protobuf from
`tensorflow/core/example/example.proto` is a wrapper of `Features`, as shown
below.

```protobuf
message Example {
  Features features = 1;
};
```

```
00000000  5b 03 00 00 00 00 00 00  21 51 51 b0 0a d8 06 0a  |[.......!QQ.....|
00000010  0f 0a 06 68 65 69 67 68  74 12 05 1a 03 0a 01 1c  |...height.......|
00000020  0a a4 06 0a 09 69 6d 61  67 65 5f 72 61 77 12 96  |.....image_raw..|
00000030  06 0a 93 06 0a 90 06 00  00 00 00 00 00 00 00 00  |................|
00000040  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
*
000000c0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 03  |................|
000000d0  12 12 12 7e 88 af 1a a6  ff f7 7f 00 00 00 00 00  |...~............|
000000e0  00 00 00 00 00 00 00 1e  24 5e 9a aa fd fd fd fd  |........$^......|
000000f0  fd e1 ac fd f2 c3 40 00  00 00 00 00 00 00 00 00  |......@.........|
00000100  00 00 31 ee fd fd fd fd  fd fd fd fd fb 5d 52 52  |..1..........]RR|
00000110  38 27 00 00 00 00 00 00  00 00 00 00 00 00 12 db  |8'..............|
00000120  fd fd fd fd fd c6 b6 f7  f1 00 00 00 00 00 00 00  |................|
00000130  00 00 00 00 00 00 00 00  00 00 00 50 9c 6b fd fd  |...........P.k..|
00000140  cd 0b 00 2b 9a 00 00 00  00 00 00 00 00 00 00 00  |...+............|
00000150  00 00 00 00 00 00 00 00  0e 01 9a fd 5a 00 00 00  |............Z...|
00000160  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000170  00 00 00 00 00 00 8b fd  be 02 00 00 00 00 00 00  |................|
00000180  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000190  00 00 0b be fd 46 00 00  00 00 00 00 00 00 00 00  |.....F..........|
000001a0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 23  |...............#|
000001b0  f1 e1 a0 6c 01 00 00 00  00 00 00 00 00 00 00 00  |...l............|
000001c0  00 00 00 00 00 00 00 00  00 00 00 00 51 f0 fd fd  |............Q...|
000001d0  77 19 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |w...............|
000001e0  00 00 00 00 00 00 00 00  00 2d ba fd fd 96 1b 00  |.........-......|
000001f0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000200  00 00 00 00 00 00 10 5d  fc fd bb 00 00 00 00 00  |.......]........|
00000210  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000220  00 00 00 00 f9 fd f9 40  00 00 00 00 00 00 00 00  |.......@........|
00000230  00 00 00 00 00 00 00 00  00 00 00 00 00 2e 82 b7  |................|
00000240  fd fd cf 02 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000250  00 00 00 00 00 00 00 27  94 e5 fd fd fd fa b6 00  |.......'........|
00000260  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00000270  00 18 72 dd fd fd fd fd  c9 4e 00 00 00 00 00 00  |..r......N......|
00000280  00 00 00 00 00 00 00 00  00 00 00 17 42 d5 fd fd  |............B...|
00000290  fd fd c6 51 02 00 00 00  00 00 00 00 00 00 00 00  |...Q............|
000002a0  00 00 00 00 00 12 ab db  fd fd fd fd c3 50 09 00  |.............P..|
000002b0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 37  |...............7|
000002c0  ac e2 fd fd fd fd f4 85  0b 00 00 00 00 00 00 00  |................|
000002d0  00 00 00 00 00 00 00 00  00 00 00 88 fd fd fd d4  |................|
000002e0  87 84 10 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
000002f0  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
*
00000340  00 00 00 00 00 00 00 0a  0e 0a 05 77 69 64 74 68  |...........width|
00000350  12 05 1a 03 0a 01 1c 0a  0e 0a 05 6c 61 62 65 6c  |...........label|
00000360  12 05 1a 03 0a 01 05 79  28 d0 d3 5b 03 00 00 00  |.......y(..[....|
00000370  00 00 00 21 51 51 b0
```
