/* Fly Brain Blackjack - 3D table view.
   Polls /state from table_server.py and draws the table, the flies, their
   cards, and a live replay of the deciding fly's brain. */

(() => {
  const POLL_MS = 250;
  const BRAIN_PLAY_MS = 1500;
  const SEAT_ANGLES = [-62, -24, 62, 22];       // degrees around the table, viewer side; last = human
  const SEAT_RADIUS = 3.15, CARD_RADIUS = 1.75;

  // ---------- scene ----------------------------------------------------
  const container = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d12);
  scene.fog = new THREE.Fog(0x0b0d12, 9, 16);

  const camera = new THREE.PerspectiveCamera(48, innerWidth / innerHeight, 0.1, 60);
  camera.position.set(0, 4.6, 6.4);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.3, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 3; controls.maxDistance = 12;

  scene.add(new THREE.HemisphereLight(0x8fa8ff, 0x1a1408, 0.55));
  const key = new THREE.SpotLight(0xfff1d6, 1.4, 30, Math.PI / 4, 0.5, 1);
  key.position.set(0, 7, 2); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);
  const fill = new THREE.PointLight(0x5ab4ff, 0.35, 20); fill.position.set(-5, 3, 4); scene.add(fill);
  const fill2 = new THREE.PointLight(0xff9a3c, 0.25, 20); fill2.position.set(5, 3, 4); scene.add(fill2);

  // table
  const felt = new THREE.Mesh(
    new THREE.CylinderGeometry(3.2, 3.2, 0.3, 64),
    new THREE.MeshStandardMaterial({ color: 0x1f6b3a, roughness: 0.95 }));
  felt.position.y = -0.15; felt.receiveShadow = true; scene.add(felt);
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(3.25, 0.14, 18, 96),
    new THREE.MeshStandardMaterial({ color: 0x4a2e1a, roughness: 0.5, metalness: 0.1 }));
  rim.rotation.x = Math.PI / 2; rim.position.y = 0.02; rim.castShadow = true; scene.add(rim);
  const pedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.6, 0.9, 2.2, 32),
    new THREE.MeshStandardMaterial({ color: 0x2a1a10, roughness: 0.7 }));
  pedestal.position.y = -1.4; scene.add(pedestal);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(12, 64),
    new THREE.MeshStandardMaterial({ color: 0x10131a, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = -2.5; floor.receiveShadow = true; scene.add(floor);

  // dealer arc marking on the felt
  const arc = new THREE.Mesh(new THREE.RingGeometry(1.15, 1.2, 64, 1, Math.PI * 1.15, Math.PI * 0.7),
    new THREE.MeshBasicMaterial({ color: 0xd9c46a, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
  arc.rotation.x = -Math.PI / 2; arc.position.y = 0.006; scene.add(arc);

  // ---------- the fly -------------------------------------------------
  function makeFly({ color = 0x3a2c25, scale = 1, dealer = false, golden = false } = {}) {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({
      color: golden ? 0xc9a23a : color, roughness: 0.55, metalness: golden ? 0.45 : 0.1 });
    const abdomenMat = new THREE.MeshStandardMaterial({
      color: golden ? 0xa8842b : 0x5b4634, roughness: 0.6, metalness: golden ? 0.4 : 0.05 });

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 16), bodyMat);
    thorax.scale.set(1, 0.9, 1.1); thorax.castShadow = true; g.add(thorax);

    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 16), abdomenMat);
    abdomen.scale.set(0.9, 0.8, 1.55); abdomen.position.set(0, -0.02, -0.42); abdomen.castShadow = true;
    g.add(abdomen);
    // stripes
    for (let i = 0; i < 3; i++) {
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(0.2 - i * 0.02, 0.018, 8, 32),
        new THREE.MeshStandardMaterial({ color: 0x1c1410, roughness: 0.8 }));
      stripe.position.set(0, -0.02, -0.30 - i * 0.14); stripe.scale.set(0.95, 0.8, 1);
      g.add(stripe);
    }

    const head = new THREE.Group(); head.position.set(0, 0.04, 0.30); g.add(head);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), bodyMat);
    skull.castShadow = true; head.add(skull);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xd12a2a, emissive: 0x4a0808, roughness: 0.25 });
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.095, 20, 14), eyeMat);
      eye.position.set(s * 0.105, 0.03, 0.07); head.add(eye);
      const glint = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8),
        new THREE.MeshBasicMaterial({ color: 0xffffff }));
      glint.position.set(s * 0.13, 0.07, 0.14); head.add(glint);
      const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.01, 0.16, 6),
        new THREE.MeshStandardMaterial({ color: 0x1c1410 }));
      ant.position.set(s * 0.05, 0.16, 0.08); ant.rotation.z = -s * 0.5; ant.rotation.x = -0.4; head.add(ant);
    }
    const mouth = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.045, 0.1, 10),
      new THREE.MeshStandardMaterial({ color: 0x2a1c14 }));
    mouth.position.set(0, -0.11, 0.06); mouth.rotation.x = 0.3; head.add(mouth);

    const legMat = new THREE.MeshStandardMaterial({ color: 0x1c1410, roughness: 0.7 });
    const legs = [];
    for (let i = 0; i < 3; i++) for (const s of [-1, 1]) {
      const leg = new THREE.Group();
      leg.position.set(s * 0.16, -0.08, 0.12 - i * 0.14);
      const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.26, 6), legMat);
      upper.position.set(s * 0.1, -0.06, 0); upper.rotation.z = s * 1.05; leg.add(upper);
      const lower = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.012, 0.26, 6), legMat);
      lower.position.set(s * 0.24, -0.2, 0); lower.rotation.z = s * 0.25; leg.add(lower);
      g.add(leg); legs.push(leg);
    }

    const wingMat = new THREE.MeshPhysicalMaterial({
      color: 0xd6e4ff, transparent: true, opacity: 0.32, roughness: 0.15, metalness: 0.2,
      side: THREE.DoubleSide, clearcoat: 1, clearcoatRoughness: 0.1 });
    const wings = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group(); pivot.position.set(s * 0.1, 0.19, -0.06);
      const w = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.24), wingMat);
      w.position.set(s * 0.31, 0, -0.05); w.rotation.x = -Math.PI / 2; w.rotation.z = s * 0.25;
      pivot.add(w); g.add(pivot); wings.push({ pivot, side: s });
    }

    if (dealer) {
      const tieMat = new THREE.MeshStandardMaterial({ color: 0xc41e3a, roughness: 0.4 });
      for (const s of [-1, 1]) {
        const half = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.12, 4), tieMat);
        half.position.set(s * 0.075, -0.08, 0.24); half.rotation.z = s * Math.PI / 2; g.add(half);
      }
      const knot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 8), tieMat);
      knot.position.set(0, -0.08, 0.25); g.add(knot);
      const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.05, 24, 1, false, 0, Math.PI),
        new THREE.MeshStandardMaterial({ color: 0x1f6b3a, transparent: true, opacity: 0.85 }));
      visor.position.set(0, 0.16, 0.08); visor.rotation.set(0.35, Math.PI, 0); head.add(visor);
    }

    g.scale.setScalar(scale);
    return { group: g, head, wings, legs, phase: Math.random() * 10, anim: "idle", animStart: 0, baseY: 0, baseRotY: 0 };
  }

  // ---------- cards ---------------------------------------------------
  const textureCache = new Map();
  function cardTexture(code) {
    if (textureCache.has(code)) return textureCache.get(code);
    const c = document.createElement("canvas"); c.width = 256; c.height = 360;
    const x = c.getContext("2d");
    const rounded = (r) => { x.beginPath(); x.moveTo(r, 0); x.arcTo(256, 0, 256, 360, r);
      x.arcTo(256, 360, 0, 360, r); x.arcTo(0, 360, 0, 0, r); x.arcTo(0, 0, 256, 0, r); x.closePath(); };
    if (code === "??") {
      rounded(22); x.fillStyle = "#1c2a55"; x.fill();
      x.strokeStyle = "#d9c46a"; x.lineWidth = 8; rounded(22); x.stroke();
      x.strokeStyle = "rgba(217,196,106,.35)"; x.lineWidth = 3;
      for (let i = -360; i < 256; i += 26) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i + 360, 360); x.stroke(); }
    } else {
      rounded(22); x.fillStyle = "#f7f4ea"; x.fill();
      const rank = code[0] === "T" ? "10" : code[0];
      const suit = { s: "♠", h: "♥", d: "♦", c: "♣" }[code[1]];
      const red = code[1] === "h" || code[1] === "d";
      x.fillStyle = red ? "#c8102e" : "#15181f";
      x.font = "bold 64px system-ui, sans-serif"; x.textBaseline = "top";
      x.fillText(rank, 18, 14); x.font = "56px system-ui, sans-serif"; x.fillText(suit, 18, 76);
      x.save(); x.translate(256, 360); x.rotate(Math.PI);
      x.font = "bold 64px system-ui, sans-serif"; x.fillText(rank, 18, 14);
      x.font = "56px system-ui, sans-serif"; x.fillText(suit, 18, 76); x.restore();
      x.font = "150px system-ui, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle";
      x.fillText(suit, 128, 190);
    }
    const t = new THREE.CanvasTexture(c); t.anisotropy = 8;
    textureCache.set(code, t); return t;
  }
  const cardGeo = new THREE.PlaneGeometry(0.5, 0.7);
  function makeCard(code) {
    const m = new THREE.Mesh(cardGeo, new THREE.MeshStandardMaterial({ map: cardTexture(code), roughness: 0.6 }));
    m.rotation.x = -Math.PI / 2; m.castShadow = true; m.receiveShadow = true; return m;
  }

  // ---------- seats ---------------------------------------------------
  const seats = SEAT_ANGLES.map((deg, i) => {
    const a = THREE.MathUtils.degToRad(deg);
    const pos = new THREE.Vector3(SEAT_RADIUS * Math.sin(a), 0.22, SEAT_RADIUS * Math.cos(a));
    const cardPos = new THREE.Vector3(CARD_RADIUS * Math.sin(a), 0.01, CARD_RADIUS * Math.cos(a));
    const cards = new THREE.Group(); cards.position.copy(cardPos); cards.rotation.y = a; scene.add(cards);
    const label = document.createElement("div"); label.className = "label";
    document.getElementById("labels").appendChild(label);
    return { index: i, angle: a, pos, cards, cardKey: "", label, fly: null, kind: "empty", brainKey: "" };
  });
  const dealer = (() => {
    const f = makeFly({ color: 0x151515, scale: 1.35, dealer: true });
    f.group.position.set(0, 0.28, -2.75); f.group.rotation.y = 0; f.baseY = 0.28; scene.add(f.group);
    const cards = new THREE.Group(); cards.position.set(0, 0.01, -1.0); cards.rotation.y = Math.PI; scene.add(cards);
    const label = document.createElement("div"); label.className = "label";
    document.getElementById("labels").appendChild(label);
    return { fly: f, cards, cardKey: "", label, pos: f.group.position };
  })();

  function setFly(seat, kind) {
    if (seat.fly) { scene.remove(seat.fly.group); seat.fly = null; }
    if (kind === "empty") return;
    const f = makeFly(kind === "human" ? { golden: true, scale: 0.95 } : { scale: 0.95 });
    f.group.position.copy(seat.pos); f.baseRotY = seat.angle + Math.PI; f.group.rotation.y = f.baseRotY;
    f.baseY = seat.pos.y;
    f.group.userData.seat = seat.index;
    scene.add(f.group); seat.fly = f;
  }

  function layCards(group, codes, key, holder) {
    if (holder.cardKey === key) return;
    holder.cardKey = key;
    while (group.children.length) group.remove(group.children[0]);
    codes.forEach((code, i) => {
      const m = makeCard(code);
      m.position.set((i - (codes.length - 1) / 2) * 0.3, 0.004 * i, 0);
      m.rotation.z = (i - (codes.length - 1) / 2) * -0.06;
      group.add(m);
    });
  }

  // ---------- animation -----------------------------------------------
  const clock = new THREE.Clock();
  function animateFly(f, t) {
    const g = f.group, dt = t - f.animStart;
    let y = f.baseY + 0.015 * Math.sin(t * 2 + f.phase), rx = 0, ry = 0, flutter = 0.06, headTilt = 0;
    switch (f.anim) {
      case "thinking": flutter = 0.55; rx = 0.18; headTilt = 0.3 * Math.sin(t * 3.2 + f.phase);
        y += 0.04 * Math.sin(t * 6); break;
      case "waiting": flutter = 0.2; headTilt = 0.25 * Math.sin(t * 2); break;
      case "hit": y += 0.18 * Math.max(0, Math.sin(Math.min(dt, 0.5) * Math.PI * 2)); rx = 0.25 * Math.exp(-dt * 2);
        flutter = 0.4 * Math.exp(-dt * 2) + 0.06; break;
      case "stand": rx = -0.22 * Math.exp(-dt * 1.5); flutter = 0.06; break;
      case "bust": case "lost": rx = 0.55; y -= 0.06; flutter = 0.0; break;
      case "won": case "blackjack": y += 0.32 * Math.abs(Math.sin(dt * 7)) * Math.exp(-dt * 0.9);
        ry = dt < 1.2 ? dt * Math.PI * 2 * 0.8 : 0; flutter = 0.7 * Math.exp(-dt * 0.8) + 0.08; break;
      case "push": headTilt = 0.35; flutter = 0.06; break;
    }
    g.position.y = y; g.rotation.x = rx; g.rotation.y = f.baseRotY + ry;
    f.head.rotation.z = headTilt;
    for (const w of f.wings) {
      const beat = flutter > 0.1 ? Math.sin(t * 44 + f.phase) * flutter : 0.05 * Math.sin(t * 3 + f.phase);
      w.pivot.rotation.z = w.side * (0.15 + beat);
      w.pivot.rotation.y = w.side * -0.15 * flutter;
      if (f.anim === "bust" || f.anim === "lost") w.pivot.rotation.z = w.side * -0.35;
    }
    for (let i = 0; i < f.legs.length; i++)
      f.legs[i].rotation.x = (f.anim === "thinking" ? 0.15 : 0.04) * Math.sin(t * 9 + i * 1.3);
  }
  function setAnim(f, anim, t) { if (f.anim !== anim) { f.anim = anim; f.animStart = t; } }

  // ---------- brain panel ---------------------------------------------
  const brainBox = document.getElementById("brain");
  const kcCanvas = document.getElementById("kc"), kcCtx = kcCanvas.getContext("2d");
  const voteCanvas = document.getElementById("votes"), voteCtx = voteCanvas.getContext("2d");
  const COLS = 96, CELL = 3, N_KC = 5177;
  let brain = null;      // { frames, votes, action, who, start, glow: Float32Array, drawn }
  let pinnedSeat = null; // seat index the viewer clicked, or null = follow whoever is thinking

  function startBrain(seat, data) {
    brain = { frames: data.kc_frames, votes: data.votes, action: data.action, who: seat.label.dataset.name,
      total: data.total, dealer: data.dealer, start: performance.now(), glow: new Float32Array(N_KC), drawn: -1 };
    brainBox.classList.remove("hidden");
    document.getElementById("brain-who").textContent = `המוח של ${brain.who}`;
    document.getElementById("brain-what").textContent =
      `${brain.total} מול ${brain.dealer === 11 ? "A" : brain.dealer}`;
  }

  function drawBrain() {
    if (!brain) return;
    const n = brain.frames.length;
    const idx = Math.min(n - 1, Math.floor((performance.now() - brain.start) / BRAIN_PLAY_MS * n));
    if (idx === brain.drawn) return;
    for (let f = brain.drawn + 1; f <= idx; f++) {
      for (let i = 0; i < N_KC; i++) brain.glow[i] *= 0.82;
      for (const k of brain.frames[f]) brain.glow[k] = 1;
    }
    brain.drawn = idx;

    kcCtx.fillStyle = "#0d1117"; kcCtx.fillRect(0, 0, kcCanvas.width, kcCanvas.height);
    let active = 0;
    const everActive = new Set();
    for (let f = 0; f <= idx; f++) for (const k of brain.frames[f]) everActive.add(k);
    for (let i = 0; i < N_KC; i++) {
      const g = brain.glow[i];
      const x = (i % COLS) * CELL, y = Math.floor(i / COLS) * CELL;
      if (g > 0.03) {
        active++;
        kcCtx.fillStyle = `rgba(90, 220, 255, ${0.25 + 0.75 * g})`;
        kcCtx.fillRect(x, y, CELL - 1, CELL - 1);
      } else if (everActive.has(i)) {
        kcCtx.fillStyle = "rgba(90, 180, 255, 0.18)";
        kcCtx.fillRect(x, y, CELL - 1, CELL - 1);
      }
    }
    document.getElementById("kc-count").textContent = everActive.size.toLocaleString();
    document.getElementById("kc-step").textContent = idx + 1;

    const v = brain.votes[idx];
    const W = voteCanvas.width, mid = W / 2, scale = 32, H = 18;
    voteCtx.clearRect(0, 0, W, voteCanvas.height);
    voteCtx.fillStyle = "#262c3a"; voteCtx.fillRect(mid - 1, 4, 2, 46);
    const rows = [["HIT", "#ff9a3c", 6], ["STAND", "#5ab4ff", 30]];
    for (const [name, color, y] of rows) {
      const val = Math.max(-3.5, Math.min(3.5, v[name] || 0));
      const len = val * scale;
      const winner = idx === n - 1 && name === brain.action;
      voteCtx.fillStyle = winner ? color : color + "99";
      if (len >= 0) voteCtx.fillRect(mid, y, len, H); else voteCtx.fillRect(mid + len, y, -len, H);
      voteCtx.fillStyle = "#e6e8ee"; voteCtx.font = "12px system-ui"; voteCtx.textBaseline = "middle";
      voteCtx.textAlign = "right"; voteCtx.fillText(name, mid - 6 - Math.max(0, -len), y + H / 2);
      voteCtx.textAlign = "left"; voteCtx.fillText(val.toFixed(2), mid + 6 + Math.max(0, len), y + H / 2);
    }
  }

  // ---------- state ---------------------------------------------------
  let state = null, myName = null, joined = false;
  const STATUS_HE = { idle: "", waiting: "תורך", thinking: "חושב...", hit: "HIT", stand: "STAND",
    bust: "נשרף", won: "ניצח", lost: "הפסיד", push: "תיקו", blackjack: "בלאקג'ק!" };

  function applyState(s) {
    state = s;
    const t = clock.getElapsedTime();
    document.getElementById("message").textContent = s.message || "";
    document.getElementById("round").textContent = `סיבוב ${s.round}` + (s.learning ? " · לומדים" : "");

    layCards(dealer.cards, s.dealer.cards, s.dealer.cards.join(","), dealer);
    dealer.label.innerHTML = `<span class="name">הדילר</span>` +
      (s.dealer.total != null ? `<span class="status">${s.dealer.total}</span>` : "");
    setAnim(dealer.fly, s.phase === "dealer" ? "thinking" : "idle", t);

    s.seats.forEach((ss, i) => {
      const seat = seats[i]; if (!seat) return;
      if (seat.kind !== ss.kind) { seat.kind = ss.kind; setFly(seat, ss.kind); }
      seat.label.dataset.name = ss.name;
      if (ss.kind === "empty") { seat.label.style.display = "none"; layCards(seat.cards, [], "", seat); return; }
      seat.label.style.display = "";
      const chips = ss.chips, cls = chips > 0 ? "pos" : chips < 0 ? "neg" : "";
      seat.label.innerHTML = `<span class="name">${ss.name}</span> ` +
        `<span class="chips ${cls}">${chips > 0 ? "+" : ""}${chips}</span>` +
        `<span class="status ${ss.status}">${STATUS_HE[ss.status] ?? ""}${ss.cards.length ? ` · ${ss.total}` : ""}</span>`;
      layCards(seat.cards, ss.cards, ss.cards.join(","), seat);
      if (seat.fly) setAnim(seat.fly, ss.status === "idle" ? "idle" : ss.status, t);

      if (ss.brain) {
        const key = `${s.round}:${ss.cards.join(",")}:${ss.brain.action}`;
        if (seat.brainKey !== key && (pinnedSeat === null || pinnedSeat === i)) {
          seat.brainKey = key; startBrain(seat, ss.brain);
        }
      }
    });

    // controls
    const humanSeat = s.seats[s.seats.length - 1];
    joined = humanSeat.kind === "human" && humanSeat.name === myName;
    document.getElementById("join-box").style.display = humanSeat.kind === "empty" && !joined ? "" : "none";
    document.getElementById("play-box").style.display = joined ? "" : "none";
    const myTurn = joined && s.human_turn;
    document.getElementById("hit").disabled = !myTurn;
    document.getElementById("stand").disabled = !myTurn;
    document.getElementById("hint").textContent = humanSeat.kind === "human" && !joined
      ? `${humanSeat.name} יושב בכיסא. גרור לסובב · לחץ על זבוב לראות את המוח שלו`
      : myTurn ? "תורך: HIT או STAND" : "גרור לסובב · גלגלת לזום · לחץ על זבוב כדי לראות את המוח שלו";
  }

  async function poll() {
    try { const r = await fetch("/state"); applyState(await r.json()); }
    catch (e) { document.getElementById("message").textContent = "השרת לא זמין"; }
    setTimeout(poll, POLL_MS);
  }

  document.getElementById("join").onclick = async () => {
    myName = (document.getElementById("name").value || "You").trim().slice(0, 16);
    await fetch("/join", { method: "POST", body: JSON.stringify({ name: myName }) });
  };
  document.getElementById("leave").onclick = async () => {
    await fetch("/leave", { method: "POST", body: "{}" }); myName = null;
  };
  document.getElementById("hit").onclick = () => fetch("/action", { method: "POST", body: JSON.stringify({ action: "HIT" }) });
  document.getElementById("stand").onclick = () => fetch("/action", { method: "POST", body: JSON.stringify({ action: "STAND" }) });
  addEventListener("keydown", (e) => {
    if (!joined || !state?.human_turn) return;
    if (e.key === "h" || e.key === "H") document.getElementById("hit").click();
    if (e.key === "s" || e.key === "S") document.getElementById("stand").click();
  });

  // click a fly to pin its brain
  const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
    mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(mouse, camera);
    const flies = seats.filter(s => s.fly && s.kind === "fly").map(s => s.fly.group);
    const hit = raycaster.intersectObjects(flies, true)[0];
    if (!hit) { pinnedSeat = null; return; }
    let o = hit.object; while (o && o.userData.seat === undefined) o = o.parent;
    if (!o) return;
    pinnedSeat = o.userData.seat;
    const ss = state?.seats[pinnedSeat];
    if (ss?.brain) { seats[pinnedSeat].brainKey = ""; startBrain(seats[pinnedSeat], ss.brain); }
  });

  // ---------- render loop ---------------------------------------------
  const tmp = new THREE.Vector3();
  function placeLabel(label, pos, lift) {
    tmp.copy(pos); tmp.y += lift; tmp.project(camera);
    label.style.left = `${(tmp.x + 1) / 2 * innerWidth}px`;
    label.style.top = `${(1 - tmp.y) / 2 * innerHeight}px`;
    label.style.opacity = tmp.z < 1 ? "1" : "0";
  }
  function render() {
    requestAnimationFrame(render);
    const t = clock.getElapsedTime();
    controls.update();
    for (const s of seats) if (s.fly) animateFly(s.fly, t);
    dealer.fly.baseRotY = 0; animateFly(dealer.fly, t);
    renderer.render(scene, camera);
    for (const s of seats) if (s.kind !== "empty") placeLabel(s.label, s.pos, 0.62);
    placeLabel(dealer.label, dealer.pos, 0.85);
    drawBrain();
  }
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  window.__table = { seats, dealer, scene, camera };
  for (const s of seats) s.fly = null;
  render(); poll();
})();
