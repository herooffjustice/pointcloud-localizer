(function () {
  const SOURCE_COLOR = 0x4fc3f7;
  const TARGET_COLOR = 0xef5350;
  const POINT_SIZE = 0.012;

  function makePoints(geometry, color, size) {
    var mat = new THREE.PointsMaterial({
      color: color,
      size: size || POINT_SIZE,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
    });
    return new THREE.Points(geometry, mat);
  }

  function arrayToGeometry(arr) {
    var geo = new THREE.BufferGeometry();
    var positions = new Float32Array(arr.length * 3);
    for (var i = 0; i < arr.length; i++) {
      positions[i * 3] = arr[i][0];
      positions[i * 3 + 1] = arr[i][1];
      positions[i * 3 + 2] = arr[i][2];
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }

  function centerPoints(arr) {
    if (!arr || arr.length === 0) return arr;
    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < arr.length; i++) {
      cx += arr[i][0]; cy += arr[i][1]; cz += arr[i][2];
    }
    cx /= arr.length; cy /= arr.length; cz /= arr.length;
    return arr.map(function (p) { return [p[0] - cx, p[1] - cy, p[2] - cz]; });
  }

  function setupScene(canvas, orbitEnabled) {
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x12121a, 1);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    camera.position.set(2, 1.5, 2);
    camera.lookAt(0, 0, 0);

    function resize() {
      var w = canvas.parentElement.clientWidth;
      var h = canvas.parentElement.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener("resize", resize);

    if (orbitEnabled !== false) {
      var isDragging = false;
      var prevX = 0, prevY = 0;
      var spherical = { radius: 3, theta: Math.PI / 4, phi: Math.PI / 3 };
      var target = new THREE.Vector3(0, 0, 0);

      function updateCamera() {
        camera.position.set(
          target.x + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta),
          target.y + spherical.radius * Math.cos(spherical.phi),
          target.z + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta)
        );
        camera.lookAt(target);
      }
      updateCamera();

      canvas.addEventListener("pointerdown", function (e) { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
      window.addEventListener("pointerup", function () { isDragging = false; });
      window.addEventListener("pointermove", function (e) {
        if (!isDragging) return;
        var dx = e.clientX - prevX;
        var dy = e.clientY - prevY;
        spherical.theta -= dx * 0.005;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.005));
        prevX = e.clientX;
        prevY = e.clientY;
        updateCamera();
      });
      canvas.addEventListener("wheel", function (e) {
        e.preventDefault();
        spherical.radius = Math.max(0.5, Math.min(10, spherical.radius + e.deltaY * 0.002));
        updateCamera();
      }, { passive: false });
    }

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    return { renderer: renderer, scene: scene, camera: camera, resize: resize };
  }

  // ---- HERO ----
  var heroCanvas = document.getElementById("hero-canvas");
  var heroSetup = setupScene(heroCanvas);
  var heroScene = heroSetup.scene;
  var heroRenderer = heroSetup.renderer;
  var heroCamera = heroSetup.camera;

  (function initHero() {
    var n = 4000;
    var geo = new THREE.BufferGeometry();
    var pos = new Float32Array(n * 3);
    var cols = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {
      var t = (i / n) * Math.PI * 2;
      var R = 1.0, r = 0.35;
      var phi = t * 3 + Math.random() * 0.5;
      var theta = t + Math.random() * 0.3;
      pos[i * 3] = (R + r * Math.cos(phi)) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi);
      pos[i * 3 + 2] = (R + r * Math.cos(phi)) * Math.sin(theta);
      var blend = Math.random();
      cols[i * 3] = 0.31 + blend * 0.62;
      cols[i * 3 + 1] = 0.76 - blend * 0.42;
      cols[i * 3 + 2] = 0.97 - blend * 0.56;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    var mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, transparent: true, opacity: 0.8, sizeAttenuation: true });
    heroScene.add(new THREE.Points(geo, mat));
    heroCamera.position.set(2.5, 1.5, 2.5);
    heroCamera.lookAt(0, 0, 0);
  })();

  var heroRotAngle = 0;
  function animateHero() {
    requestAnimationFrame(animateHero);
    heroRotAngle += 0.002;
    heroCamera.position.x = 2.8 * Math.cos(heroRotAngle);
    heroCamera.position.z = 2.8 * Math.sin(heroRotAngle);
    heroCamera.lookAt(0, 0, 0);
    heroRenderer.render(heroScene, heroCamera);
  }
  animateHero();

  // ---- LOAD DATA ----
  var regData = null;
  var algoData = null;
  var sweepData = { torus: null, box_plane: null };

  Promise.all([
    fetch("data/registration.json").then(function (r) { return r.json(); }),
    fetch("data/icp_steps.json").then(function (r) { return r.json(); }),
    fetch("data/sweep_torus.json").then(function (r) { return r.json(); }),
    fetch("data/sweep_box_plane.json").then(function (r) { return r.json(); }),
  ]).then(function (results) {
    regData = results[0];
    algoData = results[1];
    sweepData.torus = results[2];
    sweepData.box_plane = results[3];
    initRegistration();
    initAlgorithm();
    initConvergence();
    initRobustness();
  });

  // ---- REGISTRATION VIEWER ----
  var regCanvas = document.getElementById("reg-canvas");
  var regSetup = setupScene(regCanvas);
  var regScene = regSetup.scene;
  var regRenderer = regSetup.renderer;
  var regCamera = regSetup.camera;

  var sourcePoints = null;
  var targetPoints = null;
  var sourcePtpPoints = null;
  var sourcePtplPoints = null;
  var currentSource = null;

  function initRegistration() {
    var srcArr = centerPoints(regData.source);
    var tgtArr = centerPoints(regData.target);
    var srcPtpArr = centerPoints(regData.source_registered_ptp);
    var srcPtplArr = centerPoints(regData.source_registered_ptpl);

    var tgtGeo = arrayToGeometry(tgtArr);
    targetPoints = makePoints(tgtGeo, TARGET_COLOR);
    regScene.add(targetPoints);

    var srcGeo = arrayToGeometry(srcArr);
    sourcePoints = makePoints(srcGeo, SOURCE_COLOR);
    regScene.add(sourcePoints);

    var srcPtpGeo = arrayToGeometry(srcPtpArr);
    sourcePtpPoints = makePoints(srcPtpGeo, SOURCE_COLOR);
    sourcePtpPoints.visible = false;
    regScene.add(sourcePtpPoints);

    var srcPtplGeo = arrayToGeometry(srcPtplArr);
    sourcePtplPoints = makePoints(srcPtplGeo, SOURCE_COLOR);
    sourcePtplPoints.visible = false;
    regScene.add(sourcePtplPoints);

    currentSource = sourcePoints;
    updateRegStats("before");
    animateReg();
  }

  function animateReg() {
    requestAnimationFrame(animateReg);
    regRenderer.render(regScene, regCamera);
  }

  function updateRegStats(view) {
    var statsEl = document.getElementById("reg-stats");
    if (view === "before") {
      statsEl.innerHTML =
        '<div class="stat-item"><div class="stat-label">View</div><div class="stat-value" style="color:var(--text-muted)">Unaligned</div></div>' +
        '<div class="stat-item"><div class="stat-label">Method</div><div class="stat-value" style="color:var(--text-muted)">—</div></div>' +
        '<div class="stat-item"><div class="stat-label">Points</div><div class="stat-value">' + regData.source.length.toLocaleString() + '</div></div>';
    } else if (view === "ptp") {
      var r = regData.ptp_result;
      statsEl.innerHTML =
        '<div class="stat-item"><div class="stat-label">Method</div><div class="stat-value">PtP</div></div>' +
        '<div class="stat-item"><div class="stat-label">Rot Error</div><div class="stat-value">' + r.rotation_error_deg.toFixed(3) + '°</div></div>' +
        '<div class="stat-item"><div class="stat-label">Trans Error</div><div class="stat-value">' + r.translation_error_m.toFixed(5) + 'm</div></div>' +
        '<div class="stat-item"><div class="stat-label">Final RMSE</div><div class="stat-value" style="color:var(--success)">' + r.final_rmse.toFixed(5) + '</div></div>' +
        '<div class="stat-item"><div class="stat-label">Iterations</div><div class="stat-value">' + r.iterations + '</div></div>';
    } else {
      var r2 = regData.ptpl_result;
      statsEl.innerHTML =
        '<div class="stat-item"><div class="stat-label">Method</div><div class="stat-value">PtPl</div></div>' +
        '<div class="stat-item"><div class="stat-label">Rot Error</div><div class="stat-value">' + r2.rotation_error_deg.toFixed(3) + '°</div></div>' +
        '<div class="stat-item"><div class="stat-label">Trans Error</div><div class="stat-value">' + r2.translation_error_m.toFixed(5) + 'm</div></div>' +
        '<div class="stat-item"><div class="stat-label">Final RMSE</div><div class="stat-value" style="color:var(--success)">' + r2.final_rmse.toFixed(5) + '</div></div>' +
        '<div class="stat-item"><div class="stat-label">Iterations</div><div class="stat-value">' + r2.iterations + '</div></div>';
    }
  }

  document.querySelectorAll(".reg-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".reg-btn").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      var view = btn.getAttribute("data-view");
      sourcePoints.visible = view === "before";
      sourcePtpPoints.visible = view === "ptp";
      sourcePtplPoints.visible = view === "ptpl";
      currentSource = view === "before" ? sourcePoints : (view === "ptp" ? sourcePtpPoints : sourcePtplPoints);
      updateRegStats(view);
    });
  });

  // ---- ALGORITHM STEP VIEWER ----
  var algoCanvas = document.getElementById("algo-canvas");
  var algoSetup = setupScene(algoCanvas);
  var algoScene = algoSetup.scene;
  var algoRenderer = algoSetup.renderer;
  var algoCamera = algoSetup.camera;

  var algoTargetPoints = null;
  var algoSourcePoints = null;
  var algoStepGeometries = [];
  var currentAlgoStep = 0;
  var algoPlaying = false;
  var algoTimer = null;

  function initAlgorithm() {
    var tgtArr = centerPoints(algoData.target);
    var tgtGeo = arrayToGeometry(tgtArr);
    algoTargetPoints = makePoints(tgtGeo, TARGET_COLOR, 0.018);
    algoScene.add(algoTargetPoints);

    for (var i = 0; i < algoData.steps.length; i++) {
      var srcArr = centerPoints(algoData.steps[i].source);
      var srcGeo = arrayToGeometry(srcArr);
      var pts = makePoints(srcGeo, SOURCE_COLOR, 0.018);
      pts.visible = i === 0;
      algoScene.add(pts);
      algoStepGeometries.push(pts);
    }

    updateAlgoDisplay(0);
    animateAlgo();
  }

  function animateAlgo() {
    requestAnimationFrame(animateAlgo);
    algoRenderer.render(algoScene, algoCamera);
  }

  function updateAlgoDisplay(step) {
    currentAlgoStep = step;
    for (var i = 0; i < algoStepGeometries.length; i++) {
      algoStepGeometries[i].visible = i === step;
    }
    document.getElementById("algo-slider").value = step;
    document.getElementById("algo-iter-label").textContent = step;
    document.getElementById("algo-rmse-val").textContent = algoData.steps[step].rmse.toFixed(5);
  }

  document.getElementById("algo-slider").addEventListener("input", function (e) {
    var step = parseInt(e.target.value);
    updateAlgoDisplay(step);
  });

  document.getElementById("algo-play").addEventListener("click", function () {
    if (algoPlaying) {
      algoPlaying = false;
      clearInterval(algoTimer);
      this.innerHTML = "&#9654; Play";
      return;
    }
    algoPlaying = true;
    this.innerHTML = "&#9646;&#9646; Pause";
    if (currentAlgoStep >= algoData.steps.length - 1) updateAlgoDisplay(0);
    algoTimer = setInterval(function () {
      var next = currentAlgoStep + 1;
      if (next >= algoData.steps.length) {
        algoPlaying = false;
        document.getElementById("algo-play").innerHTML = "&#9654; Play";
        clearInterval(algoTimer);
        return;
      }
      updateAlgoDisplay(next);
    }, 700);
  });

  document.getElementById("algo-reset").addEventListener("click", function () {
    algoPlaying = false;
    document.getElementById("algo-play").innerHTML = "&#9654; Play";
    clearInterval(algoTimer);
    updateAlgoDisplay(0);
  });

  // ---- CONVERGENCE CHART ----
  function initConvergence() {
    var ctx = document.getElementById("convergence-chart").getContext("2d");
    var ptpHistory = regData.ptp_result.rmse_history;
    var ptplHistory = regData.ptpl_result.rmse_history;

    new Chart(ctx, {
      type: "line",
      data: {
        labels: Array.from({ length: Math.max(ptpHistory.length, ptplHistory.length) }, function (_, i) { return i + 1; }),
        datasets: [
          {
            label: "Point-to-Point",
            data: ptpHistory,
            borderColor: "#4fc3f7",
            backgroundColor: "rgba(79, 195, 247, 0.1)",
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: "#4fc3f7",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Point-to-Plane",
            data: ptplHistory,
            borderColor: "#b388ff",
            backgroundColor: "rgba(179, 136, 255, 0.1)",
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: "#b388ff",
            tension: 0.3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            labels: { color: "#e4e4ef", font: { family: "'Inter', sans-serif", size: 12 } },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Iteration", color: "#8888a0" },
            ticks: { color: "#8888a0" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
          y: {
            title: { display: true, text: "RMSE", color: "#8888a0" },
            ticks: { color: "#8888a0" },
            grid: { color: "rgba(255,255,255,0.05)" },
          },
        },
      },
    });
  }

  // ---- ROBUSTNESS SWEEP ----
  var currentSweepMesh = "torus";
  var currentMetric = "rotation_error_deg";

  function initRobustness() {
    renderSweep();
    renderSweepTable();

    document.querySelectorAll(".sweep-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".sweep-btn").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        currentSweepMesh = btn.getAttribute("data-mesh");
        renderSweep();
        renderSweepTable();
      });
    });

    document.querySelectorAll(".metric-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".metric-btn").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        currentMetric = btn.getAttribute("data-metric");
        renderSweep();
      });
    });
  }

  function getMetricData(metric) {
    var data = sweepData[currentSweepMesh];
    if (!data) return null;
    if (metric === "rotation_error_deg") return data.rotation_errors;
    if (metric === "translation_error_m") return data.translation_errors;
    return data.rmse_values;
  }

  function heatmapColor(val, min, max) {
    if (max === min) return "rgba(108, 99, 255, 0.4)";
    var t = (val - min) / (max - min);
    var r = Math.round(40 + t * 215);
    var g = Math.round(180 - t * 140);
    var b = Math.round(255 - t * 200);
    return "rgba(" + r + "," + g + "," + b + ",0.85)";
  }

  function renderSweep() {
    var data = sweepData[currentSweepMesh];
    if (!data) return;
    var metricData = getMetricData(currentMetric);
    var wrap = document.getElementById("heatmap-wrap");
    var nNoise = data.noise_levels.length;
    var nMisalign = data.misalignment_labels.length;

    var allVals = [];
    for (var i = 0; i < nNoise; i++)
      for (var j = 0; j < nMisalign; j++)
        allVals.push(metricData[i][j]);
    var minVal = Math.min.apply(null, allVals);
    var maxVal = Math.max.apply(null, allVals);

    var html = '<div class="heatmap-grid" style="grid-template-columns: 60px repeat(' + nMisalign + ', 1fr);">';
    html += '<div class="heatmap-label"></div>';
    for (var j = 0; j < nMisalign; j++) {
      html += '<div class="heatmap-label">' + data.misalignment_labels[j] + '</div>';
    }
    for (var i = 0; i < nNoise; i++) {
      html += '<div class="heatmap-label">σ=' + data.noise_levels[i] + '</div>';
      for (var j = 0; j < nMisalign; j++) {
        var val = metricData[i][j];
        var formatted = currentMetric === "rotation_error_deg" ? val.toFixed(2) + "°"
          : currentMetric === "translation_error_m" ? val.toFixed(4) + "m"
          : val.toFixed(4);
        html += '<div class="heatmap-cell" style="background:' + heatmapColor(val, minVal, maxVal) + '">' + formatted + '</div>';
      }
    }
    html += '</div>';
    wrap.innerHTML = html;
  }

  function renderSweepTable() {
    var data = sweepData[currentSweepMesh];
    if (!data) return;
    var thead = document.querySelector("#sweep-table thead tr");
    var tbody = document.querySelector("#sweep-table tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    var cols = ["Noise σ (m)", "Misalignment", "Rot Error (°)", "Trans Error (m)", "Final RMSE", "Iterations"];
    cols.forEach(function (c) {
      var th = document.createElement("th");
      th.textContent = c;
      thead.appendChild(th);
    });

    data.rows.forEach(function (r) {
      var tr = document.createElement("tr");
      var cells = [
        r.noise_sigma,
        r.max_rotation_deg + "° / " + r.max_translation_m + "m",
        r.rotation_error_deg.toFixed(3),
        r.translation_error_m.toFixed(5),
        r.final_rmse.toFixed(5),
        r.iterations,
      ];
      cells.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
})();
