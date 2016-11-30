# tensorflow-special-octo-spoon

## Writing TensorFlow Records

It is possible to write TFRecords using the `tf.python_io.TFRecordWriter`
class, which can be used in Python `with` blocks, and whose purpose is to write
records to a TFRecords file.

A `tf.python_io.TFRecordWriter` object is initialized with the filename of the
output file. We want to call the `tf.python_io.TFRecordWriter.write` method on
(TODO(brendan): refer to Features protobuf format)

### <a name="protobuf"></a>Google Protobuf Format

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

##### Encoding

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
