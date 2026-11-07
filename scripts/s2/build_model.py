"""
Step 2 (M1): build step2/model/anymal_d.xml, the MuJoCo ANYmal D model, from the description's URDF.

- Source: ANYbotics/anymal_d_simple_description, urdf/anymal.urdf (commit and sha256 in step2/assets/manifest.json).
- Bodies: links joined by fixed joints are fused into one body (as Isaac Lab's USD import and MuJoCo's own
  `fusestatic` do). The moving bodies are base (free joint), and per leg HIP, THIGH and SHANK. The two revolute
  joints of the inspection payload are fixed (they are not driven in the walking task).
- Inertia: each body's mass, centre of mass and full inertia tensor are summed from its links at full double
  precision and diagonalised with numpy; principal moments and axes are written in full (MuJoCo's own XML writer
  rounds to 6 digits, and its diagonalisation of `fullinertia` is accurate only to about 1e-6).
- Joints: the 12 leg hinges with the URDF's axes and limits, written in the data's joint order for the actuators.
- Collision shapes, simplified from the URDF's inline primitives (all visual meshes dropped):
  box trunk (the URDF's base box); per leg a capsule thigh between the centres of the thigh's two end cylinders,
  radius half the thigh box's width; a capsule shank along the URDF's shank tube; a sphere foot (URDF radius).
  Robot shapes collide with the ground only (no self-collision).
- Ground plane; time step 0.005 s; one torque motor per joint, limited to Isaac Lab's actuator effort limit 80 N m.
- Solver options follow MuJoCo Menagerie's ANYmal C (cone elliptic, impratio 100). The four calibration constants
  (friction, contact time constant, armature, damping) get placeholder values here; scripts/s2/mj_env.py sets them.

Run: ./py-mj scripts/s2/build_model.py   (writes the XML; prints its sha256)
"""
import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
URDF = os.path.join(AIR, "step2", "assets", "anymal_d", "urdf", "anymal.urdf")
OUT = os.path.join(AIR, "step2", "model", "anymal_d.xml")

LEGS = ("LF", "LH", "RF", "RH")
JOINT_ORDER = [f"{leg}_{j}" for j in ("HAA", "HFE", "KFE") for leg in LEGS]   # the data's order (rwm_data.py:52)
PAYLOAD_JOINTS = ("inspection_payload_mount_to_pan", "inspection_payload_pan_to_tilt")
EFFORT_LIMIT = 80.0          # Isaac Lab ANYMAL_D actuator_effort_limit (step2/notes/M0_facts.md a)
DT = 0.005
# placeholders; the environment overrides them (calibration grid in STEP2_PLAN.md)
FRICTION, TIMECONST, ARMATURE, DAMPING = 0.8, 0.02, 0.0, 0.0


# ------------------------------------------------------------------ geometry helpers
def rpy_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_T(el):
    T = np.eye(4)
    if el is not None:
        xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
        T[:3, :3] = rpy_R(*rpy)
        T[:3, 3] = xyz
    return T


def R_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def f(v):
    return " ".join(repr(float(x)) for x in np.atleast_1d(v))


# ------------------------------------------------------------------ URDF
def parse(path=URDF):
    root = ET.parse(path).getroot()
    links = {}
    for l in root.findall("link"):
        d = {"inertial": None, "coll": []}
        it = l.find("inertial")
        if it is not None:
            a = {k: float(v) for k, v in it.find("inertia").attrib.items()}
            I = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]],
                          [a["ixz"], a["iyz"], a["izz"]]])
            d["inertial"] = (float(it.find("mass").get("value")), origin_T(it.find("origin")), I)
        for c in l.findall("collision"):
            g = c.find("geometry")[0]
            d["coll"].append((g.tag, {k: float(v) if g.tag != "box" else [float(x) for x in v.split()]
                                      for k, v in g.attrib.items()}, origin_T(c.find("origin"))))
        links[l.get("name")] = d
    joints = []
    for j in root.findall("joint"):
        lim = j.find("limit")
        ax = j.find("axis")
        joints.append({
            "name": j.get("name"),
            "type": "fixed" if j.get("name") in PAYLOAD_JOINTS else j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "T": origin_T(j.find("origin")),
            "axis": np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None,
            "lower": float(lim.get("lower")) if lim is not None else None,
            "upper": float(lim.get("upper")) if lim is not None else None,
        })
    return links, joints


def fuse(links, joints):
    """Moving bodies with their fused links: {body: {parent, T (in parent body), joint, links {link: T_body_link}}}."""
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    child_links = {j["child"] for j in joints}
    roots = [l for l in links if l not in child_links]
    assert roots == ["base"], roots
    bodies = {"base": {"parent": None, "T": np.eye(4), "joint": None, "links": {"base": np.eye(4)}}}
    owner = {"base": ("base", np.eye(4))}
    stack = ["base"]
    while stack:
        p = stack.pop()
        ob, Tp = owner[p]
        for j in children.get(p, []):
            c = j["child"]
            Tc = Tp @ j["T"]
            if j["type"] == "fixed":
                owner[c] = (ob, Tc)
                bodies[ob]["links"][c] = Tc
            elif j["type"] == "revolute":
                bodies[c] = {"parent": ob, "T": Tc, "joint": j, "links": {c: np.eye(4)}}
                owner[c] = (c, np.eye(4))
            else:
                raise ValueError(j["type"])
            stack.append(c)
    return bodies, owner


def body_inertial(body, links):
    """Total mass, centre of mass and inertia about it, in the body frame (double precision)."""
    ms, cs, Is = [], [], []
    for ln, T in body["links"].items():
        it = links[ln]["inertial"]
        if it is None:
            continue
        m, To, I = it
        R = T[:3, :3] @ To[:3, :3]
        ms.append(m)
        cs.append((T @ To)[:3, 3])
        Is.append(R @ I @ R.T)
    M = sum(ms)
    c = sum(m * x for m, x in zip(ms, cs)) / M
    I = np.zeros((3, 3))
    for m, x, Ii in zip(ms, cs, Is):
        r = x - c
        I += Ii + m * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c, I


def coll(owner, links, link, tag, k=0):
    """The k-th collision primitive of type tag on link, as (attrs, T in the owning body's frame)."""
    prims = [(a, T) for t, a, T in links[link]["coll"] if t == tag]
    a, T = prims[k]
    return a, owner[link][1] @ T


# ------------------------------------------------------------------ MJCF
def geoms_for(name, owner, links):
    """Simplified collision shapes (see module docstring) for moving body `name`."""
    out = []
    if name == "base":
        a, T = coll(owner, links, "base", "box")
        out.append(f'<geom name="trunk" type="box" size="{f(np.array(a["size"]) / 2)}" pos="{f(T[:3, 3])}" '
                   f'quat="{f(R_quat(T[:3, :3]))}"/>')
        return out
    leg, part = name.split("_")
    if part == "THIGH":
        tl = f"{leg}_thigh_fixed"
        (_, T0), (_, T1) = coll(owner, links, tl, "cylinder", 0), coll(owner, links, tl, "cylinder", 1)
        box, _ = coll(owner, links, tl, "box")
        r = box["size"][0] / 2
        out.append(f'<geom name="{leg}_thigh" type="capsule" size="{f(r)}" '
                   f'fromto="{f(T0[:3, 3])} {f(T1[:3, 3])}"/>')
    elif part == "SHANK":
        fl = f"{leg}_FOOT"
        cyl, Tc = coll(owner, links, fl, "cylinder")
        r, h = cyl["radius"], cyl["length"] / 2
        ax = Tc[:3, 2]
        p0, p1 = Tc[:3, 3] - (h - r) * ax, Tc[:3, 3] + (h - r) * ax
        out.append(f'<geom name="{leg}_shank" type="capsule" size="{f(r)}" fromto="{f(p0)} {f(p1)}"/>')
        sph, Ts = coll(owner, links, fl, "sphere")
        out.append(f'<geom name="{leg}_foot" type="sphere" size="{f(sph["radius"])}" pos="{f(Ts[:3, 3])}"/>')
    return out


def body_xml(name, bodies, owner, links, ind):
    b = bodies[name]
    pad = "  " * ind
    M, c, I = body_inertial(b, links)
    attrs = f'name="{name}"'
    if b["parent"] is not None:
        attrs += f' pos="{f(b["T"][:3, 3])}" quat="{f(R_quat(b["T"][:3, :3]))}"'
    else:
        attrs += ' pos="0 0 0.6"'
    s = [f"{pad}<body {attrs}>"]
    ev, V = np.linalg.eigh(I)               # principal axes here: MuJoCo's own diagonalisation of
    if np.linalg.det(V) < 0:                 # `fullinertia` is accurate only to about 1e-6 for the base
        V[:, 0] = -V[:, 0]
    s.append(f'{pad}  <inertial pos="{f(c)}" quat="{f(R_quat(V))}" mass="{f(M)}" diaginertia="{f(ev)}"/>')
    if b["joint"] is None:
        s.append(f'{pad}  <freejoint name="root"/>')
    else:
        j = b["joint"]
        s.append(f'{pad}  <joint name="{j["name"]}" type="hinge" axis="{f(j["axis"])}" '
                 f'range="{f([j["lower"], j["upper"]])}"/>')
    s += [pad + "  " + g for g in geoms_for(name, owner, links)]
    for cn, cb in bodies.items():
        if cb["parent"] == name:
            s += body_xml(cn, bodies, owner, links, ind + 1)
    s.append(f"{pad}</body>")
    return s


def build(path=URDF):
    links, joints = parse(path)
    bodies, owner = fuse(links, joints)
    lines = [
        '<mujoco model="anymal_d">',
        '  <!-- Generated by scripts/s2/build_model.py from anymal_d_simple_description urdf/anymal.urdf.',
        '       Hand-built physics model for AIR 2027 Step 2; not the data\'s own simulator. -->',
        '  <compiler angle="radian" autolimits="true"/>',
        f'  <option timestep="{DT}" cone="elliptic" impratio="100"/>',
        '  <default>',
        f'    <joint limited="true" damping="{DAMPING}" armature="{ARMATURE}"/>',
        f'    <geom contype="1" conaffinity="0" condim="3" friction="{FRICTION} 0.005 0.0001" '
        f'solref="{TIMECONST} 1"/>',
        f'    <motor ctrllimited="true" ctrlrange="{-EFFORT_LIMIT} {EFFORT_LIMIT}" gear="1"/>',
        '  </default>',
        '  <worldbody>',
        f'    <geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="1"/>',
    ]
    lines += body_xml("base", bodies, owner, links, 2)
    lines += ['  </worldbody>', '  <actuator>']
    lines += [f'    <motor name="{j}" joint="{j}"/>' for j in JOINT_ORDER]
    lines += ['  </actuator>', '</mujoco>', '']
    return "\n".join(lines)


def main():
    xml = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(xml)
    print(OUT, hashlib.sha256(xml.encode()).hexdigest())


if __name__ == "__main__":
    main()
