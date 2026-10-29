# bluegrass jam splitter

Drop a long jam recording into the **Jam Inbox** folder in Google Drive. Within about half an hour,
the individual songs show up in **Jam Tracks/&lt;recording name&gt;/** as
`01 - Salt Creek.mp3`, `02 - Blackberry Blossom.mp3`, … with the chat and dead time between
songs cut out. Each folder also gets a `tracklist.txt` that shows where each song sits in the
original and why it was given that title. The original recording is moved to
`Jam Inbox/processed`.

## How it works

`jamsplit.py` runs on GitHub Actions every 15 minutes (`.github/workflows/process.yml`):

1. **rclone** downloads anything new from `Jam Inbox`.
2. An AudioSet classifier (AST) scores every 10 s window for *speech* vs. *bluegrass music*.
3. **Whisper** transcribes the recording: the chat between songs (often names the tune) and the sung lyrics.
4. **Claude** reads the timeline and transcript, then decides the song boundaries and titles.
5. **ffmpeg** cuts the songs out of the original file without re-encoding, then rclone uploads them.

If the Claude step fails, the classifier's boundaries are used and the tracks are named "Unknown tune N".

## One-time setup (laptop)

1. Install rclone: <https://rclone.org/install/>.
2. Connect it to Google Drive. A browser window opens for you to log in:
   ```bash
   rclone config create gdrive drive scope=drive
   ```
3. Print the config. The output includes a login token, so keep it private:
   ```bash
   cat "$(rclone config file | tail -1)"
   ```
4. On GitHub, open this repo → **Settings → Secrets and variables → Actions → New repository secret**, and add:
   - `RCLONE_CONF`: the whole output of step 3.
   - `ANTHROPIC_API_KEY`: your Anthropic API key.
5. Open the **Actions** tab → *Process jam recordings* → **Run workflow** to test it straight away.

To use different folder names, add repository *variables* `JAM_INBOX` / `JAM_OUTPUT`.

## Phone (Android)

- **Manual:** in your recorder app or Files, tap Share → **Drive**, then choose the **Jam Inbox** folder.
- **Automatic:** install *Autosync for Google Drive* (MetaCtrl). Set up a one-way "Upload only" sync from your
  recorder's folder to `Jam Inbox`. New recordings then upload by themselves, ideally on Wi‑Fi only.

## Faster pickup (optional)

GitHub's free schedule can run hours late. `drive-trigger.gs` is a Google Apps Script that
checks Jam Inbox every 5 minutes and starts the GitHub job immediately:

1. Create a GitHub token at <https://github.com/settings/personal-access-tokens/new>. Give it access
   to only this repository, with the permission **Actions: Read and write**.
2. At <https://script.google.com>, create a new project, paste in `drive-trigger.gs`, and put your
   token into it.
3. Select `setup` and press **Run**, then approve the permissions prompt.

## Running locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ANTHROPIC_API_KEY=... python jamsplit.py local recording.mp3 --outdir out/
```

## Notes

- Processing a 2.5 h recording takes about 30–60 min on a GitHub runner. Private repos get 2,000 free
  Actions minutes a month; public repos are unlimited. Recordings never get committed to the repo.
- GitHub pauses scheduled workflows in repos with no commits for 60 days. If that happens, click
  "Enable workflow" in the Actions tab.
- Tunes that nobody names out loud and that have no lyrics can't be identified, so they are called "Unknown tune N".
