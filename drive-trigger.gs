// Google Apps Script: start the GitHub job as soon as a recording lands in Jam Inbox.
// GitHub's own 15-minute schedule is often delayed by hours; this checks every 5 minutes.
//
// Setup (script.google.com → New project → paste this file):
//   1. Put your GitHub token below (fine-grained, this repo only, "Actions: Read and write").
//   2. Pick `setup` in the function dropdown and press Run; approve the permissions prompt.

const GITHUB_TOKEN = 'PASTE_GITHUB_TOKEN_HERE';
const REPO = 'jacobcd52/bluegrass';
const INBOX_FOLDER_ID = '19z0NxKqLBvPhgF3tanizOAFBtAiCBNMM'; // Drive folder "Jam Inbox"
const RETRY_HOURS = 3; // start another run if the same files are still waiting after this long

function setup() {
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));
  ScriptApp.newTrigger('checkInbox').timeBased().everyMinutes(5).create();
  checkInbox();
}

function checkInbox() {
  const props = PropertiesService.getScriptProperties();
  const ids = [];
  const files = DriveApp.getFolderById(INBOX_FOLDER_ID).getFiles(); // files only, not processed/
  while (files.hasNext()) ids.push(files.next().getId());
  if (!ids.length) {
    props.deleteProperty('seen');
    return;
  }
  const key = ids.sort().join(',');
  const seen = JSON.parse(props.getProperty('seen') || '{}');
  if (seen.key === key && Date.now() - seen.at < RETRY_HOURS * 3600e3) return;

  const res = UrlFetchApp.fetch(
    `https://api.github.com/repos/${REPO}/actions/workflows/process.yml/dispatches`, {
      method: 'post',
      contentType: 'application/json',
      headers: {Authorization: 'Bearer ' + GITHUB_TOKEN, Accept: 'application/vnd.github+json'},
      payload: JSON.stringify({ref: 'main'}),
      muteHttpExceptions: true,
    });
  if (res.getResponseCode() === 204) {
    props.setProperty('seen', JSON.stringify({key, at: Date.now()}));
    console.log(`started GitHub job for ${ids.length} file(s)`);
  } else {
    console.log(`GitHub said ${res.getResponseCode()}: ${res.getContentText()}`);
  }
}
