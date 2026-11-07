#!/bin/zsh
# Step 2 fetch job (run by the launcher, never by a session).
# Downloads/installs only into step2/assets/ and .pylib-mj/. Resumable: skips finished items.
emulate -L zsh
set -u
here=${0:A:h:h:h}
cd "$here" || exit 1
A=step2/assets
mkdir -p "$A/actuator" .pylib-mj results/s2 .cache/logs
PYBIN="${here:h}/.venv-rwm311/bin/python"

DESC_URL=https://github.com/ANYbotics/anymal_d_simple_description
DESC_COMMIT=d5bf4dc3a81af81a16b9598fc15c98e339f45a33
# ANYdrive network: file name from Isaac Lab v3.0.0 anymal.py (ISAACLAB_NUCLEUS_DIR/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt).
# The asset root is read at runtime by Isaac Lab, so several candidate roots are tried; all are NVIDIA's public asset server.
ACT_REL="Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"
ACT_ROOTS=(
  "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
  "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets"
)
MJ_VER=3.10.0
PKGS=(mujoco==$MJ_VER absl-py==2.1.0 etils==1.7.0 glfw==2.7.0 pyopengl==3.1.7
      importlib_resources==6.4.0 zipp==3.19.2)

progress() { print -r -- "{\"job\":\"fetch\",\"step\":\"$1\",\"time\":\"$(date +%FT%T)\"}" > results/s2/progress.json; }
declare -A STATUS

# 1. robot description
progress description
if [[ ! -d $A/anymal_d/.git ]]; then
  git clone --quiet "$DESC_URL" "$A/anymal_d" && git -C "$A/anymal_d" checkout --quiet "$DESC_COMMIT" \
    && STATUS[description]=ok || STATUS[description]=FAILED
else
  git -C "$A/anymal_d" checkout --quiet "$DESC_COMMIT" && STATUS[description]=ok || STATUS[description]=FAILED
fi
DESC_HEAD=$(git -C "$A/anymal_d" rev-parse HEAD 2>/dev/null || echo none)
[[ $DESC_HEAD == $DESC_COMMIT ]] || STATUS[description]=FAILED
print "description: ${STATUS[description]} commit=$DESC_HEAD"

# 2. ANYdrive network (failure is recorded, not fatal: M1 then uses the PD fallback)
progress actuator
ACT=$A/actuator/anydrive_3_lstm_jit.pt
ACT_SRC=none
if [[ ! -s $ACT ]]; then
  for r in $ACT_ROOTS; do
    if curl -fsSL --max-time 300 -o "$ACT.part" "$r/$ACT_REL" && [[ -s $ACT.part ]]; then
      mv "$ACT.part" "$ACT"; ACT_SRC="$r/$ACT_REL"; break
    fi
    rm -f "$ACT.part"
  done
else ACT_SRC=existing
fi
if [[ -s $ACT ]]; then
  ACT_SHA=$(shasum -a 256 "$ACT" | cut -d' ' -f1); STATUS[actuator]=ok
else
  ACT_SHA=none; STATUS[actuator]="FAILED (M1 uses the PD fallback)"
fi
print "actuator: ${STATUS[actuator]} sha256=$ACT_SHA"

# 3. MuJoCo and helpers into .pylib-mj
progress pip
for p in $PKGS; do
  name=${p%%==*}
  "$PYBIN" -m pip install --quiet --disable-pip-version-check --target .pylib-mj --no-deps --upgrade "$p" \
    >> .cache/logs/fetch_pip.log 2>&1 && STATUS[$name]=ok || STATUS[$name]=FAILED
  print "pip $p: ${STATUS[$name]}"
done

# 4. ./py-mj
progress py-mj
cat > py-mj <<'EOF'
#!/bin/zsh
# ./py-mj <args>: as ./py, with .pylib-mj (MuJoCo and helpers) also on PYTHONPATH.
emulate -L zsh
here=${0:A:h}
export PYTHONDONTWRITEBYTECODE=1
export PDM_ROOT="${here:h}"
export PYTHONPATH="$here/scripts${PYTHONPATH:+:$PYTHONPATH}"
[[ -d "$here/.pylib" ]] && export PYTHONPATH="$PYTHONPATH:$here/.pylib"
[[ -d "$here/.pylib-mj" ]] && export PYTHONPATH="$PYTHONPATH:$here/.pylib-mj"
export MPLCONFIGDIR="$here/.cache/matplotlib"
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "${here:h}/.venv-rwm311/bin/python" "$@"
EOF
chmod +x py-mj
CHECK=$(./py-mj -c "import mujoco, numpy, torch; m = mujoco.MjModel.from_xml_string('<mujoco/>'); print(mujoco.__version__, numpy.__version__)" 2>&1 | tail -1)
print "check: $CHECK"
[[ $CHECK == "$MJ_VER 1.26.4" ]] && STATUS[check]=ok || STATUS[check]="FAILED: $CHECK"

# 5. manifest
progress manifest
export DESC_URL DESC_COMMIT DESC_HEAD ACT_SRC ACT_SHA MJ_VER CHECK
export S_DESC=${STATUS[description]} S_ACT=${STATUS[actuator]} S_CHECK=${STATUS[check]}
export PKG_LIST="${(j: :)PKGS}"
export PKG_STATUS="${(j: :)${(@k)STATUS}}"
"$PYBIN" -I - <<'EOF'
import json, os, hashlib, datetime, glob
e = os.environ
pk = {}
for p in e["PKG_LIST"].split():
    n, v = p.split("==")
    pk[n] = {"version": v, "source": "PyPI", "installed_into": ".pylib-mj"}
# wheel hashes from pip's dist-info RECORD are not sha256 of the wheel; record installed dist-info presence
for n in pk:
    d = glob.glob(".pylib-mj/%s-*.dist-info" % n.replace("-", "_").lower()) + glob.glob(".pylib-mj/%s-*.dist-info" % n)
    pk[n]["dist_info"] = sorted(set(os.path.basename(x) for x in d))
m = {
  "written": datetime.datetime.now().isoformat(timespec="seconds"),
  "description": {"url": e["DESC_URL"], "commit": e["DESC_COMMIT"], "checked_out": e["DESC_HEAD"],
                  "licence": "BSD 3-Clause (ANYbotics AG, 2023)", "status": e["S_DESC"],
                  "urdf": "urdf/anymal.urdf (collision shapes are inline primitives)"},
  "actuator": {"file": "anydrive_3_lstm_jit.pt", "source": e["ACT_SRC"], "sha256": e["ACT_SHA"],
               "status": e["S_ACT"], "committed": False, "in_supplement": False,
               "note": "Isaac Lab docstring: no public ANYmal-D net exists; the ANYmal-C ANYdrive net is used"},
  "mujoco": {"version": e["MJ_VER"], "import_check": e["CHECK"], "status": e["S_CHECK"]},
  "packages": pk,
}
json.dump(m, open("step2/assets/manifest.json", "w"), indent=2)
print("manifest written")
EOF
progress done
print "manifest: step2/assets/manifest.json"
