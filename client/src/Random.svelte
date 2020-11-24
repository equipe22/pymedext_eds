<script>
  let rand = -1;

  function getRand() {
    fetch("./rand")
      .then(d => d.text())
      .then(d => (rand = d));
  }

  function handleClick() {
    getRand();
  }
</script>

<h1>The count is {rand}</h1>
<button on:click={handleClick}>get random number</button>
