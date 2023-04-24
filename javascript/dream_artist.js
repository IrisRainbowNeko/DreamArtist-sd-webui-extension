
function start_training_dreamartist(){
    gradioApp().querySelector('#da_error').innerHTML=''

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('da_output'), gradioApp().getElementById('da_gallery'), function(){}, function(progress){
        gradioApp().getElementById('da_progress').innerHTML = progress.textinfo
    })

    var res = args_to_array(arguments)

    res[0] = id

    return res
}

onUiUpdate(function(){
    check_progressbar('da', 'da_progressbar', 'da_progress_span', '', 'da_interrupt', 'da_preview', 'da_gallery')
})