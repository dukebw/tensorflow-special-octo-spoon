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
not human readable, and a \*.proto file is needed to describe the structure of
the data so that it can be parsed (or deserialized).

To read more about the motivation and history behind protobufs, see the
[Protocol Buffers Overview](https://developers.google.com/protocol-buffers/docs/overview).

#### `Features` Example

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

Note that the `syntax = "proto3";` line in the feature.proto file indicates
that the proto3 (as opposed to proto2) version of the protobuf standard is used.

`BytesList`, `FloatList`, `Int64List`, `Feature`, and `Features` are all
message types.

Each field in each message type definition has a unique numbered tag. E.g. the
field `FloatList float_list` in message type `Feature` has a tag of 2. Tags in
the range [0-15] are encoded in one byte.

The `repeated` keyword indicates that a message may have zero or more of this
field. E.g. `BytesList` is made up of a sequence of zero or more `bytes`.

TODO(brendan): What is the meaning of [packed = true], and why is it not
present in `BytesList`?
