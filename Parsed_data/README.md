# Parsed data folder

This folder contain our code for parsing the Maestro dataset with two different approaches. One approach is the main parsing method we used with our final model, which is the dense encoding approach. Each song contained 3 columns: the note played, how long it was played, and how long before the next note was played from the current note. The matrix is arranged top to bottom or order of the song.
The other is a time-based array where a matrix was formed with the key lines up on the y-axis and x-axis being time. Each key is given either a 0 for not played or 1 for played during that time column.

* All_Maestro_Parsed.npy - a saved array using our main parsing approach of the maestro data set (200k examples)
* Start_Maestro_Parsed.npy - a saved array using our main parsing approach of the start of songs of the maestro data set (15k examples)
* All_pianoroll_v1.npy.zip - a saved array using the alternative method **Not Used**
* Parsed MIDI Files.ipynb - a notebook used for making All_Maestro_Parsed.npy or Start_Maestro_Parsed.npy
* Parsed MIDI Files. Test.ipynb -helps explain Parsed MIDI Files.ipynb
* Parse MIDI Files-Generating_note_timestemp_array.ipynb - a notebook used for making All_pianoroll_v1.npy **Not Used**

**The main parsing code is Parse MIDI Files.ipynb and the main saved array is All_Maestro_Parsed.npy**