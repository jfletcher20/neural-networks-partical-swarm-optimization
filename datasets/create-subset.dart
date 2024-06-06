import 'dart:io';

void main() {
  // list the files in the current directory

  double max = 0, min = 0;
  String filename = "PENGUINS";
  final file = File('$filename.csv');
  final lines = file.readAsLinesSync();

  for (final line in lines) {
    final values = line.split(',');

    for (final value in values) {
      final parsedValue = double.tryParse(value);

      if (parsedValue != null && parsedValue < min)
        min = parsedValue;
      else if (parsedValue != null && parsedValue > max) max = parsedValue;
    }
  }

  if (max != min) {
    print("Domain: [$min, $max]");
  } else {
    print('No valid numbers found in the file.');
  }

  // now get all the rows which contain the text "Adelie", all the rows which contain the text "Chinstrap", and all the rows which contain the text "Gentoo"
  var adelie = lines.where((line) => line.contains('Adelie'));
  var chinstrap = lines.where((line) => line.contains('Chinstrap'));
  var gentoo = lines.where((line) => line.contains('Gentoo'));

  // check which has the least amount of entries, and reduce the other lists to the same length
  min = adelie.length.toDouble();
  if (chinstrap.length < min) min = chinstrap.length.toDouble();
  if (gentoo.length < min) min = gentoo.length.toDouble();

  adelie = adelie.take(min.toInt());
  chinstrap = chinstrap.take(min.toInt());
  gentoo = gentoo.take(min.toInt());

  // now overwrite the csv file with each list combined such that they're interlaced in pattern Adelie\nChinstrap\nGentoo\nAdelie\nChinstrap\nGentoo\n...
  var combined = <String>[];
  for (var i = 0; i < min.toInt(); i++) {
    combined.add(adelie.elementAt(i));
    combined.add(chinstrap.elementAt(i));
    combined.add(gentoo.elementAt(i));
  }

  file.writeAsStringSync(combined.join('\n').trimRight());
  print('File has been shuffled and saved.');
  // check the amount of occurrences of each species and output it
  var adelieCount = file.readAsLinesSync().where((line) => line.contains('Adelie')).length;
  var chinstrapCount = file.readAsLinesSync().where((line) => line.contains('Chinstrap')).length;
  var gentooCount = file.readAsLinesSync().where((line) => line.contains('Gentoo')).length;

  print('Adelie: $adelieCount');
  print('Chinstrap: $chinstrapCount');
  print('Gentoo: $gentooCount');
}
