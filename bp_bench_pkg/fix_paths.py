import glob, os, re
base = os.getcwd()
compiler = base + '/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang'
upim = base + '/uPIMulator'
for f in glob.glob('uPIMulator/sdk/**/CMakeLists.txt', recursive=True):
    with open(f) as fh:
        s = fh.read()
    s = re.sub(r'[^\s"]+/dpu-upmem-dpurte-clang', compiler, s)
    s = re.sub(r'[^\s"]+/uPIMulator(?=[^/]|$)', upim, s)
    with open(f, 'w') as fh:
        fh.write(s)
    print('Fixed: ' + f)
