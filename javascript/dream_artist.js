
function start_training_dreamartist(){
    requestProgress('da')
    gradioApp().querySelector('#da_error').innerHTML=''

    return args_to_array(arguments)
}

onUiUpdate(function(){
    check_progressbar('da', 'da_progressbar', 'da_progress_span', '', 'da_interrupt', 'da_preview', 'da_gallery')
})