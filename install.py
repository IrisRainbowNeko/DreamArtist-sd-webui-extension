import launch

def get_tabs(data:str, idx:int):
    ln_idx=data.rindex('\n', 0, idx)+1
    return data[ln_idx:idx]

def add_text(data, text, ref, head, ln_callback=None):
    idx_ti_output = data.index(ref)
    ln_idx = data.index('\n', idx_ti_output) + 1 if ln_callback is None else ln_callback(data, idx_ti_output)
    tabs = get_tabs(data, idx_ti_output)
    if data.find(text, idx_ti_output)==-1:
        data = data[:ln_idx] + tabs + text + data[ln_idx:]
    return data

# hooks for show infos
with open('modules/ui.py', 'r', encoding='utf8') as f:
    ui_file_data = f.read()
    with open('./log.txt', 'w+', encoding='utf8') as f2:
        f2.write(ui_file_data)

    ui_file_data=add_text(ui_file_data, 'shared.ti_output = ti_output\n', 'ti_output = gr.', 'shared.')
    ui_file_data=add_text(ui_file_data, 'shared.ti_outcome = ti_outcome\n', 'ti_outcome = gr.', 'shared.')

    if ui_file_data.find('params.dream_artist_trigger()') == -1:
        idx_click = ui_file_data.index('create_embedding.click(')
        tabs = get_tabs(ui_file_data, idx_click)
        ui_file_data = ui_file_data[:idx_click] + 'params.dream_artist_trigger()\n' \
                       + tabs + ui_file_data[idx_click:]


with open('modules/ui.py', 'w', encoding='utf8') as f:
    f.write(ui_file_data)
    print('hook ui.py over')

if not launch.is_installed("scikit_learn"):
    launch.run_pip("install scikit_learn", "requirements for scikit_learn")