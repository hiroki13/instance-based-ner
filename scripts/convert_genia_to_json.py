# coding=utf-8
import argparse
import codecs
import ujson


def load(filename):
  with codecs.open(filename, mode="r", encoding="utf-8") as f:
    words, tags = [], []
    for line in f:
      line = line.lstrip().rstrip()
      if len(line) == 0:
        if len(words) != 0:
          yield words, tags
          words, tags = [], []
      else:
        line = line.split("\t")
        words.append(line[0])
        tags.append(line[1:])


def write_json(filename, data):
  with codecs.open(filename, mode="w", encoding="utf-8") as f:
    ujson.dump(data, f, ensure_ascii=False)


def remove_duplicate_sents(sents):
  new_sents = []
  for i, (words1, tags1) in enumerate(sents):
    for (words2, _) in sents[i + 1:]:
      if words1 == words2:
        break
    else:
      new_sents.append((words1, tags1))
  return new_sents


def bio2span(labels):
  spans = []
  span = []
  for w_i, label in enumerate(labels):
    if label.startswith('B-'):
      if span:
        spans.append(span)
      span = [label[2:], w_i, w_i]
    elif label.startswith('I-'):
      if span:
        if label[2:] == span[0]:
          span[2] = w_i
        else:
          spans.append(span)
          span = [label[2:], w_i, w_i]
      else:
        span = [label[2:], w_i, w_i]
    else:
      if span:
        spans.append(span)
      span = []
  if span:
    spans.append(span)
  return spans


def main(argv):
  sents = list(load(argv.input_file))
  print("Sents: %d" % len(sents))
  if argv.remove_duplicates:
    sents = remove_duplicate_sents(sents)
    print("Sents (removed duplicates): %d" % len(sents))

  data = []
  n_sents = 0
  n_words = 0
  n_spans = 0
  for words, multi_layered_labels in sents:
    spans = []
    for bio_labels in zip(*multi_layered_labels):
      spans += bio2span(bio_labels)
    data.append({"sent_id": n_sents,
                 "words": words,
                 "spans": spans})
    n_sents += 1
    n_words += len(words)
    n_spans += len(spans)

  if argv.output_file.endswith(".json"):
    path = argv.output_file
  else:
    path = argv.output_file + ".json"
  write_json(path, data)
  print("Sents:%d\tWords:%d\tEntities:%d" % (n_sents, n_words, n_spans))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='SCRIPT')
  parser.add_argument('--input_file',
                      help='path to genia corpus')
  parser.add_argument('--output_file',
                      default="output",
                      help='output file name')
  parser.add_argument('--remove_duplicates',
                      action='store_true',
                      default=False,
                      help='remove duplicates')
  main(parser.parse_args())
