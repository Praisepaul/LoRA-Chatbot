/** ===================== MAIN ===================== **/
function runCollectAllIOMsToCspmSheet() {
  const auth = falconAuth_();
  const ioms = falconCollectIOMs(auth);
  buildIOMCsvForCspmSheet_(ioms);
  Logger.log(`✅ CSPM IOMs exported to sheet: CSPM (${ioms.length} records)`);
}

/** ===================== COLLECT IOMs ===================== **/
function falconCollectIOMs(auth) {
  const { base, token } = auth;

  // Keep this conservative to avoid URL length issues on entity lookups
  const QUERY_LIMIT = 100;
  const ENTITY_BATCH = 25;

  let ids = [];
  let nextToken = null;
  const filter = "severity:[0,1]";

  do {
    let url = `${base}/detects/queries/iom/v2?limit=${QUERY_LIMIT}&filter=${encodeURIComponent(filter)}`;
    if (nextToken) {
      url += `&next_token=${encodeURIComponent(nextToken)}`;
    }

    const resp = UrlFetchApp.fetch(url, {
      method: "get",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/json"
      },
      muteHttpExceptions: true
    });

    const code = resp.getResponseCode();
    const body = resp.getContentText();

    if (code !== 200) {
      Logger.log(`Queries error (${code}): ${body}`);
      throw new Error(`IOM query failed with HTTP ${code}`);
    }

    const json = JSON.parse(body);
    const pageIds = json.resources || [];
    ids.push(...pageIds);

    nextToken = json.meta && json.meta.pagination
      ? json.meta.pagination.next_token || null
      : null;

    Logger.log(`Fetched ${pageIds.length} IDs this page; ${ids.length} total so far...`);
  } while (nextToken);

  Logger.log(`Total IDs fetched: ${ids.length}`);

  if (!ids.length) return [];

  let results = [];

  for (let i = 0; i < ids.length; i += ENTITY_BATCH) {
    const chunk = ids.slice(i, i + ENTITY_BATCH);
    const idParams = chunk.map(id => `ids=${encodeURIComponent(id)}`).join("&");
    const url = `${base}/detects/entities/iom/v2?${idParams}`;

    Logger.log(`Fetching entity batch ${Math.floor(i / ENTITY_BATCH) + 1} of ${Math.ceil(ids.length / ENTITY_BATCH)}...`);

    const resp = UrlFetchApp.fetch(url, {
      method: "get",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/json"
      },
      muteHttpExceptions: true
    });

    const code = resp.getResponseCode();
    const body = resp.getContentText();

    if (code === 200) {
      const data = JSON.parse(body);
      results.push(...(data.resources || []));
    } else {
      Logger.log(`Entities error (${code}): ${body}`);
    }
  }

  Logger.log(`Total entity records fetched: ${results.length}`);

  // Optional post-filter:
  // Only keep High/Critical in case API behavior changes or query is broadened later.
  results = results.filter(r => {
    const sev = String(r.severity || "").toLowerCase();
    return sev === "high" || sev === "critical";
  });

  Logger.log(`Final IOMs after filter: ${results.length}`);
  return results;
}

/** ===================== BUILD CSV ===================== **/
function buildIOMCsvForCspmSheet_(ioms) {
  const header = [
    "Cloud Provider",
    "Account Name",
    "Service",
    "Severity",
    "Region",
    "Resource ID Type",
    "Policy Statement",
    "Policy Type",
    "Status",
    "Scan Time",
    "Report Date Time",
    "Resource Create Time",
    "Resource URL",
    "Finding Summary",
    "Tags"
  ];

  const rows = [header];

  ioms.forEach(iom => {
    rows.push([
      iom.cloud_provider || "",
      iom.account_name || "",
      iom.resource_id_type || "",
      iom.service || "",
      iom.region || "",
      iom.policy_statement || "",
      iom.severity || "",
      iom.status || "",
      iom.scan_time || "",
      iom.report_date_time || "",
      iom.resource_create_time || "",
      iom.resource_url || "",
      flattenFinding_(iom.finding || iom.findings || {}),
      flattenTags_(iom.tags)
    ]);
  });

  writeIOMsToCspmSheet_(rows);
}

/** ===================== HELPERS ===================== **/
function flattenFinding_(finding) {
  if (!finding) return "";

  if (Array.isArray(finding)) {
    return finding.map(item => flattenFinding_(item)).filter(Boolean).join(" | ");
  }

  if (typeof finding === "object") {
    return Object.keys(finding)
      .map(key => `${key}: ${finding[key]}`)
      .join(" | ");
  }

  return String(finding);
}

function flattenTags_(tags) {
  if (!tags) return "";

  if (Array.isArray(tags)) {
    return tags.map(t => {
      if (typeof t === "object") {
        return Object.keys(t).map(k => `${k}=${t[k]}`).join(",");
      }
      return String(t);
    }).join(" | ");
  }

  if (typeof tags === "object") {
    return Object.keys(tags).map(k => `${k}=${tags[k]}`).join(" | ");
  }

  return String(tags);
}

/** ===================== WRITE TO SHEET ===================== **/
function writeIOMsToCspmSheet_(rows) {
  const ss = SpreadsheetApp.openById("1k8hpS9pTnfDZtFNqpWhGg-GNvmf37bbubsMjOOO_uSY");
  let sheet = ss.getSheetByName("CSPM");
  if (!sheet) sheet = ss.insertSheet("CSPM");

  sheet.clearContents();
  sheet.getRange(1, 1, rows.length, rows[0].length).setValues(rows);
  sheet.getRange(1, 1, 1, rows[0].length).setFontWeight("bold");
  sheet.autoResizeColumns(1, rows[0].length);
}