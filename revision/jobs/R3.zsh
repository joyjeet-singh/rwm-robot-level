#!/bin/zsh
# Job R3 (REVISION_PLAN.md R3 step 8): study FT's fine-tuning runs, then scoring (prereg/PREREG_FT.md).
# U30F for the 15 parents, then U50F (results/ft_probe.json: projected 3.95 h <= 8 h, so U50F runs).
# A run whose record exists is skipped. Exits non-zero on the first failure. Finally ./py scripts/ft_eval.py.
emulate -L zsh
setopt err_return pipe_fail
cd "${0:A:h:h:h}" || exit 1
[[ -f prereg/PREREG_FT.md && -x ./py ]] || { print "R3 job: not in AIR 2027"; exit 1; }
mkdir -p .cache/logs
log=.cache/logs/job_R3_$(date +%Y%m%d-%H%M%S).log

rid() {  # rid <a-b> <seed>
  if [[ $1 == 1-8 ]]; then print "armA_seed$2"; else print "x3_armA_ho${1/-/}_seed$2"; fi
}

for lev in u30 u50; do
  level=0.${lev#u}
  for fold in 1-8 0-2 3-4 5-6 7-9; do
    for seed in 0 1 2; do
      id="ft_${lev}_$(rid $fold $seed)"
      if [[ -f results/raw/ft/$id.json ]]; then
        print "$id: skipped (record exists)"
        continue
      fi
      t0=$SECONDS
      if ./py scripts/ft_unstructured.py $id $fold $seed $level 500 >> $log 2>&1; then
        print "$id: done in $(( SECONDS - t0 )) s"
      else
        print "$id: FAILED (see $log)"
        exit 1
      fi
    done
  done
done

print "ft_eval: scoring"
if ./py scripts/ft_eval.py >> $log 2>&1; then
  print "ft_eval: done"
else
  print "ft_eval: FAILED (see $log)"
  exit 1
fi
