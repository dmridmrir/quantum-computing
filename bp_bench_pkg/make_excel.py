import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'BF-UF Bench'

headers = ['p', 'CPU us/dec', 'alg3 1-DPU us/dec', 'alg3.v 1-DPU us/dec',
           'alg3 Thr (dec/s)', 'alg3.v Thr (dec/s)', 'alg3 speedup', 'alg3.v speedup']
data = [
    [0.001, 1.525, 64.577, 66.018, 39642305, 38777440, 7.6, 7.4],
    [0.003, 2.194, 120.707, 112.128, 21208410, 22831077, 5.8, 6.3],
    [0.005, 3.909, 195.594, 175.391, 13088359, 14595973, 6.4, 7.1],
    [0.007, 6.170, 278.443, 248.092, 9193974, 10318772, 7.1, 8.0],
    [0.009, 8.692, 402.183, 356.479, 6365261, 7181349, 6.9, 7.8],
]

header_fill = PatternFill('solid', fgColor='1F4E79')
header_font = Font(bold=True, color='FFFFFF')
thin = Side(style='thin')
border = Border(left=thin, right=thin, top=thin, bottom=thin)

for col, h in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=h)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal='center')
    cell.border = border

row_fills = [PatternFill('solid', fgColor='DEEAF1'), PatternFill('solid', fgColor='FFFFFF')]
for row_i, row in enumerate(data, 2):
    for col_i, val in enumerate(row, 1):
        cell = ws.cell(row=row_i, column=col_i, value=val)
        cell.fill = row_fills[row_i % 2]
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
        if col_i in (1, 2, 3, 4):
            cell.number_format = '0.000'
        elif col_i in (5, 6):
            cell.number_format = '#,##0'
        elif col_i in (7, 8):
            cell.number_format = '0.0'

col_widths = [8, 14, 20, 22, 18, 18, 15, 16]
for i, w in enumerate(col_widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w

info = [
    ('Code', '[[144,12,12]] Bivariate Bicycle QLDPC'),
    ('Decoder', 'BF-UF Algorithm 3 / Algorithm 3-var'),
    ('Ref', 'Loebl et al., arXiv:2407.15988 (2024)'),
    ('CPU', 'Intel Xeon E5-2686 v4 @ 2.30GHz (8 cores)'),
    ('DPU', 'UPMEM PIM @ 350 MHz, K=11 tasklets/DPU, 100 exec (uPIMulator)'),
    ('Compiler', 'gcc -O2 (CPU) | dpu-upmem-dpurte-clang (DPU)'),
    ('Trials', '1100 syndromes per p (warmup 2000), 100 exec x K=11 tasklets'),
    ('Thr', '2560 DPUs x 11 tasklets @ 350 MHz'),
    ('Speedup', '2560-DPU Thr / (8-core CPU Thr)'),
]
for i, (k, v) in enumerate(info, 8):
    ws.cell(row=i, column=1, value=k).font = Font(bold=True)
    ws.cell(row=i, column=2, value=v)

wb.save(r'C:\Users\wjdrm\bp_bench_pkg\bfuf_bench_result.xlsx')
print('saved')
