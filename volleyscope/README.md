# volleyscope

Offline volleyball match analysis from a single camera. No internet, no paid
services: OpenCV + NumPy on a CPU.

You give it video of a set; it finds the court, follows the players and the
ball, works out each contact (serve, reception, set, attack, block, dig, free
ball) and what came of it, and produces per-player and team statistics,
tendencies, a first-person replay of lost points, and PDF sheets.

**Read "What has and has not been verified" before trusting any number it
prints about your own match.**

## What has and has not been verified

- Everything was built and tested on **computer-generated clips** with known
  answers (stick-figure players, one clean gym, a tripod camera behind the end
  line). The tests check that the program recovers those answers.
- **On the one piece of real footage tried, it did not produce usable
  results.** See "Result on real footage" below. Treat every figure in the
  table after it as a statement about generated clips only.
- **PyTorch was not installed in the build environment**, so the optional
  PyTorch person detector (`--detector torch`) is written but has never been
  executed. The tested path uses no neural networks at all.
- The interactive `calibrate` window was not tested (no display). The
  `--export` variant, which only saves a frame, shares its file handling.

### Result on real footage

Three short clips of one real match were tried (screen recordings of a
stream, 2068x1158, fixed camera behind the end line about 1.9 m off the
floor, glossy tile-pattern floor, both teams in mostly white jerseys,
spectators and officials around the court).

- **Automatic court detection failed.** The seams of the floor tiles read as
  dozens of lines. The first attempt returned a confidently wrong court; it
  now refuses instead (a fit that no real camera could produce is rejected).
  A hand calibration from four line intersections worked and looked right.
- **Player tracking failed.** With the camera that low, the far half of the
  court is about 35 pixels tall at processing size and players stand in front
  of one another almost all the time. The detector returned merged groups,
  floor reflections and spectators rather than individual players. The two
  teams' jerseys could not be told apart by colour. One jersey number was
  "read", and it was wrong.
- **Ball tracking partly worked.** Serve arcs starting behind the end line
  were found; the contacts after them were mostly wrong, and the ball leaves
  the top of the frame on high passes.
- **Net result: no trustworthy stats.** 6 "rallies" and 15 "contacts" were
  reported from a 38-second clip, nearly all flagged low-confidence or with
  no player attached. The flags did their job; the numbers are not usable.

What this footage needs that this build does not have: a trained person
detector and ball detector (the `--detector torch` hook is where one would
go; it is untested), or a camera mounted much higher.

### Measured on generated clips

One 73-second, 8-rally clip, rendered at 960x540 with a panning and zooming
camera, written to an MP4 file and analysed from that file (so video
compression is included):

| What | Result |
|---|---|
| Court position error | 2 cm average, 23 cm worst (pan + zoom test clip) |
| Players followed | 95% of player-frames, 18 cm average position error |
| Identity purity (share of a track's frames on one true player) | 88% |
| Jersey numbers | 4 read correctly, 0 read wrongly, 11 tracks unread |
| Contacts found | 38 of 39, plus 2 that did not happen |
| Contact type right | 36 of the 38 |
| Right player on the contact (after labelling tracks by hand) | 32 of 38; wrong player on 5 |
| Attack outcome right | 8 of 9 |
| Pass grade right | 10 of 10 |
| Rally winner right | 8 of 8 |
| The one block in the clip | missed (scored as a hitting error; the point went to the right team) |

With perfect ball and player positions fed in directly (no vision), the
classification rules reproduce the simulator's labels on 296 of 296 contacts
and 60 of 60 rally winners; shot type (line/cross/tip/roll) on 60 of 67
attacks. So on generated clips the remaining errors come from the vision
stages, not the volleyball logic.

Run the tests yourself: `python -m pytest` (about 6 minutes; `-m "not slow"`
skips the rendered-video tests and takes under 2 minutes).

## Tagging by hand (the route that works on real footage today)

Automatic tracking did not work on the real footage tried (see above). The
dependable route is for a person to mark the touches; the statistics,
tendencies and PDF sheets then run on those tags.

1. Open `tagger/tagger.html` in a browser (double-click it; it works offline)
   and choose your video with **Open video**. Type the two team names.
2. For each touch: pause, type the **jersey number**, press the **touch key**
   (`S` serve, `R` reception, `E` set, `A` attack, `B` block, `D` dig, `F` free
   ball), then the **result** (`K` kill / ace / stuff, `X` error, `O` in play,
   `L` blocked) and, where asked, a digit for the detail (pass grade 0-3, or
   shot 1 line, 2 cross, 3 tip, 4 roll). `Enter` saves it. `Tab` switches team.
3. Optional: click the small court to mark where an attack was hit from (the
   hitter's own half) and where it landed (the other half). Line/cross,
   deep-corner and "hits from the left" figures need these clicks.
4. A point-ending result (kill, ace, error, stuff, a grade-0 pass) closes the
   rally and starts the next with the winner serving. If the rally ended on a
   touch you did not tag, press **Point** for the team that won it.
5. **Save tags file**, then:

```
vball import-tags set1_tags.json --out set1
vball player set1 --team "My Team" --number 7 --pdf me.pdf
vball team set1 --team "My Team" --pdf team.pdf
vball tendencies set1 --team "My Team"
```

Notes:

- You may tag every touch or only some (for example only your own). Only
  tagged touches are counted, so a percentage describes the touches you
  tagged. Side-out and points-lost figures need every rally to have a server
  and a point awarded, even if you tag nothing else in it.
- "My team" is always drawn at the bottom of the court diagram, whichever end
  you played from. Tag each set to its own file and list them together.
- Work is kept in the browser between sessions, but save the tags file as
  your real copy.
- Pass grades are whatever you enter; the geometric grading described below
  applies only to automatic analysis.
- The first-person replay needs player positions, which tags do not record;
  it is not available for hand-tagged sets.
- Tested: the tag-to-stats path; the page's logic under Node; and the page
  itself in Chrome with one of the real clips (video loaded and played, a
  rally was tagged by keyboard and court clicks, the saved file imported
  cleanly). Not tested in Safari or Firefox. If your browser shows a black
  frame for your video, convert the file to H.264 MP4.

## Install

Python 3.10+.

```
pip install numpy opencv-python scipy pyyaml matplotlib reportlab
pip install pytest pypdf        # only to run the tests
pip install -e .                # optional: gives you the `vball` command
```

Without the last line, use `python -m vball.cli` wherever this file says `vball`.

Check the install on generated footage (this proves it runs, nothing more):

```
vball demo --out demo_out
```

## Using it on your own match

Film from **behind an end line, as high as you can, on a tripod**, with the
whole court in frame. One video per set is simplest, because teams change ends.

```
# 1. analyse (slow: roughly real time or slower on a laptop)
vball analyze set1.mp4 --out set1 --near "My Team" --far "Opponent"

# 2. look at set1/tracks_review.png: one picture per tracked player with the
#    number the program read ("?" if none). Write what it got wrong or missed
#    into a corrections file (format below), then:
vball relabel set1 --corrections corrections.yaml

# 3. your own numbers, by jersey number
vball player set1 --team "My Team" --number 7 --pdf me.pdf

# 4. team report and tendencies (several sets can be listed)
vball team set1 set2 --team "My Team" --pdf team.pdf
vball tendencies set1 set2 --team "My Team"

# 5. first-person replay of a lost point, and the placement simulator
vball pov set1 --team "My Team"                       # lists lost points
vball pov set1 --team "My Team" --rally 12 --out pov  # one image per defender
vball simulate pov/rally12_scenario.json --place "2,3;4.5,2;7,3;2,7;4.5,7.5;7,7"
```

`--near` is the team on the camera's side of the net in that video. For the
second set, swap the names.

### The corrections file

Jersey numbers are the part most likely to need your help. `tracks_review.png`
shows every track as `T<id>`.

```yaml
numbers: {3: 7, 5: 12}     # track 3 is number 7, track 5 is number 12
merge: [[3, 9]]            # tracks 3 and 9 are the same player
teams: {4: far}            # track 4 was put on the wrong side
ignore: [11]               # track 11 is the referee
```

`relabel` applies it and re-scores every rally without re-reading the video.

### What the stats mean

- **Kill %** = kills / attack attempts. **Attack efficiency** = (kills -
  errors - blocked) / attempts.
- **Reception grade 3/2/1/0** is geometric, not a coach's eye: it is read from
  where the next touch (the set) is played. Within 1.6 m of the net = 3,
  within 3.7 m = 2, further = 1, no second touch = 0.
- **In system / out of system** follows the pass: grade 2 or 3 is in system.
- **Sets ending in a kill** is the "sets to point" ratio.
- **Rotation 1-6** counts from whatever rotation each team started the set
  in; the program does not know which player is your setter.
- Every figure shows its sample. **no data** means nothing was seen (never
  shown as 0). **low sample** means fewer than 5 events. **uncertain** means
  more than a quarter of the events behind it were low-confidence. Contacts
  whose player number is unknown go to an "unidentified" line and are never
  added to anyone's totals.
- A tendency is only stated with at least 8 observations; below that it
  prints `insufficient data (n=...)`. One set is not enough for most of them.

### How contacts are classified

Rules, in plain terms. These are approximations of what a scorer does.

- A rally starts with a contact at or behind an end line: the **serve**.
- The first touch by the other team is a **reception** (after a serve) or a
  **dig** (after anything else). The second touch on the same side is a
  **set**. A touch that sends the ball over is an **attack** if the ball was
  above 2.0 m when hit, otherwise a **free ball**.
- A touch at the net within 0.6 s of an attack is a **block**. If the ball is
  never seen at the block but comes straight back to the attacker's side, a
  block is *inferred* when an opposing player was standing at the net there;
  otherwise it is scored as an attack error.
- Attack shots: slower than 7.7 m/s is a **tip** (lands within 3.5 m of the
  net) or a **roll shot** (deeper); faster is **line** or **cross** by angle.
- The point goes against the side the ball lands on, or against the last team
  to touch a ball that lands out. Whoever serves next is used as a check; a
  disagreement lowers the rally's confidence.

## Accuracy limits, in plain language

### Things that depend on the camera

- **Camera position.** Built for a camera behind the end line, raised. From
  there, players line up behind one another and hide each other; the program
  then places hidden players from the top of their jersey and an assumed
  height, which can be off by half a metre or more along the line of sight.
  A low camera makes all of this worse: positions far from the camera are
  stretched, so a small pixel error becomes metres.
- **Side-on cameras** are handled by the court finder, but "near team" then
  means the team on the left, and the far sideline's players hide behind the
  near ones. Not tested beyond court detection.
- **A hand-held or moving camera** breaks the player detector, which assumes
  the camera only turns and zooms from one spot. Use a tripod.
- **Depth of the ball.** One camera cannot see how far away the ball is. The
  program solves for it from the curve gravity gives the flight. That works
  well for long flights (5-15 cm on generated clips) and badly for short,
  fast ones travelling straight away from the camera, which is exactly what a
  hard attack looks like from behind the end line. Those are pinned to the
  previous touch and to the spot where the ball is seen to land; if the
  landing is not seen, the landing position can be metres off and the rally
  is flagged.
- **Air drag and spin are ignored.** Float serves and topspin do not follow a
  clean parabola; expect landing points and ball speeds to be biased on real
  footage.
- **Resolution.** Jersey numbers need digits at least about 12 pixels tall.
  At 960-pixel width from behind the end line that means only the players
  closest to the camera, and only single digits reliably. The reader runs at
  the processing resolution; film in at least 1080p and raise `--width` if
  numbers are unread (slower).
- **Line calls.** A ball within 35 cm of a line is marked a close call and
  the rally is low-confidence. The program has no way to make a real line
  call.

### Court

- Lines must be white-ish and brighter than the floor. Gyms with many
  coloured lines for other sports will often defeat automatic detection: use
  manual calibration.
- At least two sidelines and two cross-court lines must be visible to start.
  After that it survives partial views by following image motion, and says so
  (court status `propagated` or `weak`, lower confidence).
- A sudden change covering most of the picture is treated as a camera cut and
  the court is searched for again.

### Players

- Detection is by difference from an empty-court background plus floor
  colour. It fails where a jersey is the same colour as the floor, on shiny
  floors with strong reflections or shadows, and for anyone standing still in
  the same place for the whole video in front of a non-floor background.
- Splitting players who overlap in the image relies on the two teams wearing
  clearly different jersey colours. A libero in a third colour, patterned
  jerseys, or a jersey close to skin tone will confuse it.
- **Identity swaps happen** when teammates bunch together (at the net, in
  serve receive), because teammates look the same. On the generated clip 12%
  of track time was on the wrong player. Jersey numbers repair swaps only for
  players whose numbers are legible. A swap moves contacts from one player's
  sheet to another's and is not detectable from the output alone.
- After a camera cut or time off-frame, a player is matched back by jersey
  number if read, otherwise by where they were and how they look. Two
  teammates returning at once can come back swapped.
- No more than six tracks per side are kept at once; a substitute coming on
  is treated as the player who left unless the number is read.
- Jumps: feet leaving the floor look like a dash away from the camera. Short
  out-and-back excursions are removed, so a real two-step dash and back can be
  flattened too.
- The jersey reader matches digit shapes against fonts available offline. It
  was 79/80 on a font it had not seen, but those are clean computer fonts;
  real block numerals on wrinkled fabric are untested. It is designed to say
  nothing rather than guess, and it still made occasional wrong reads on
  generated clips before voting.

### Ball and contacts

- The ball is found as a small round moving blob, and by colour once the
  program has learned what colour the ball is. A ball the colour of a jersey,
  the floor or the wall will be lost. Shoes or sleeves in the ball's colour
  create false ball candidates; slow ones are filtered out.
- A contact is where the ball's path changes. A touch that barely changes the
  path (a block touch, a brush) is not detected. Hence: **touch-outs are
  scored as attack errors**, and block touches are under-counted.
- **Blocks are the weakest event.** A stuff block is usually inferred, not
  seen, and can be mistaken for a hit into the net (or the reverse).
- The contact is given to the nearest tracked player on that side within
  2.5 m. When two players are equally close (a setter and a hitter crossing)
  it can pick the wrong one; such contacts are flagged.
- Contacts hidden behind players are placed by joining the flights either
  side, and are flagged `contact_not_seen_directly`.
- If the serve is not seen, the rally still counts but its first contact is
  `unknown` and the rotation count is marked uncertain from then on.
- Set type (quick, shoot, back set), serve type and hitter approach are not
  recognised.

### Stats, tendencies, replay

- Stats are only as right as the contacts under them. Two missed contacts in
  a rally can change who is charged with the error.
- Rotation numbers are counted from serve changes. One missed side-out shifts
  every later rotation by one; affected rallies carry a `?`.
- "Broken play" and "out of system" follow the geometric pass grade, not a
  coach's judgement of the play.
- Opponent blocking is counted from who is *standing* at the net in front of
  the hitter, not who jumped.
- The first-person replay draws where tracked players stood, from eye height,
  looking at the hitter. It is a diagram, not the camera's view. "Could have
  reached it" uses fixed numbers for every player: 0.3 s reaction, 4.5 m/s,
  1 m lunge. Defenders the tracker lost at that moment are missing from the
  picture, and the diagnosis says how many.
- The coaching notes are rules of thumb applied to your numbers (for
  example: efficiency under +0.10, or more than 65% of attacks the same
  shot). They are prompts for you and your coach, not a judgement.

## When things go wrong

**The court is not found** (`The court could not be found in this video`):
calibrate by hand on one frame where the court is clear.

```
vball calibrate set1.mp4 --frame 300 --export frame300.png
```

Open the image, read off the pixel position of at least four known court
points (corners, ends of the centre line; the court is 9 x 18 m, origin at
the near-left corner, net at y = 9) and write:

```json
{"frame": 300, "points": [
  {"image": [212, 446], "court": [0, 0]},  {"image": [748, 446], "court": [9, 0]},
  {"image": [655, 217], "court": [9, 18]}, {"image": [305, 217], "court": [0, 18]}]}
```

then `vball analyze set1.mp4 --out set1 --calibration calib.json ...`. If the
camera never moves, add `--static-camera` so that calibration is used for
every frame instead of being re-fitted to the lines.
(`vball calibrate set1.mp4 --frame 300` opens a click-to-mark window instead;
that window is untested.)

**Few or no rallies found:** check `info.json` (`ball_colour_learned`,
`n_flights`). A ball that blends into the background is the usual cause.

**Stats for my number are empty:** your track probably has no number. Check
`tracks_review.png` and add it to the corrections file.

## Layout

```
vball/court      line detection, court finding, per-frame homography
vball/players    background model, detection, tracking, jersey reader, corrections
vball/ball       ball tracklets, 3D flights, contacts, outcomes, rallies
vball/stats      player and team statistics
vball/tendencies setter / hitter / defence tendencies, coaching notes
vball/pov        lost-point reconstruction, first-person render, simulator
vball/report     PDF sheets
vball/synth      generated matches with ground truth (used by the tests)
tests/           unit tests per module + end-to-end tests on rendered clips
```

Output of `analyze`: `tracks.json` (positions in metres per player),
`rallies.json` (every contact with type, outcome, confidence, flags),
`court.json`, `balls.json`, `info.json`, `tracks_review.png`.
